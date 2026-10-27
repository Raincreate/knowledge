"""
PDF转MD节点
使用MinerU将PDF文档转换为MarkDown格式
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Tuple

import requests

from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError


class PdfToMdNode(BaseNode):
    """
    PDF转MD节点
    调用MinerU命令将PDF转换为Markdown,支持实时输出转换日志
    """

    name = "pdf_to_md_node"

    def process(self, state: T) -> T:
        """
        执行pdf转换
        :param state: 图状态
        :return: 更新后的图状态（包含：md_path）
        """

        # 1.对参数校验
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        # 2.利用MinerU解析pdf为md--本地模型调用
        # processed_code = self._execute_mineru(import_file_path, file_dir_path)
        processed_code = self._execute_mineru_online(import_file_path, file_dir_path)
        if processed_code != 0:
            raise PdfConversionError("pdf转md失败", self.name)

        # 3.获取md的path
        md_path = self._get_md_paths(import_file_path, file_dir_path)

        # 4.更新state字典的md_path字段
        state["md_path"] = md_path

        # 5.返回更新后的state
        return state

    def _validate_state_inputs_path(self, state) -> Tuple[Path, Path]:
        """
        验证输入路径
        :param state: 该节点接收到的状态
        :return: (import_file_path_obj,file_dir_path_obj)元组
        """

        self.log_step("step1", "对状态的路径输入参数进行校验")

        # 1.获取pdf文件路径
        import_file_path = state.get("import_file_path", "")

        # 2.获取解析后的输出目录
        file_dir = state.get("file_dir", "")

        # 3.判断文件是否存在
        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)

        # 4.用Path标准化路径
        import_file_path_obj = Path(import_file_path)

        # 5.校验是一个真实的路径
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不是真实存在路径", self.name)

        # 6.判断输出目录是否为空,就取输入文件的目录作为输出目录
        if not file_dir:
            file_dir = import_file_path_obj.parent

        # 7.用Path标准化路径
        file_dir_obj = Path(file_dir)

        # 8.打印日志
        self.log_step(f"解析的pdf文件完成路径：{import_file_path}")
        self.log_step(f"解析后MD文件输出目录：{file_dir}")

        return (import_file_path_obj, file_dir_obj)  # 返回元组

    def _execute_mineru(self, import_file_path: Path, file_dir_path: Path) -> int:
        """
        执行mineru命令将pdf转换为md
        :param import_file_path: 解析的pdf文件全路径
        :param file_dir_path: 解析后md的存储目录
        :return: 命令执行结果状态码  0 表示成功
        """
        self.log_step("step2", "执行MinerU解析 PDF to MarkDown")

        # 1.构建命令
        # cmd = [ "mineru","-p",str(import_file_path),"-o",str(file_dir_path),"--source","local"]
        cmd = ["mineru", "-p", str(import_file_path), "-o", str(file_dir_path), "--backend", "pipeline", "--device", "cpu"]

        process_start_time = time.time()

        # 2.执行命令 （子进程执行命令）
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,  # 子进程执行命令的日志输出回流
            stderr=subprocess.STDOUT,  # 子进程执行命令的错误日志输出回流
            errors="replace",  # 遇到乱码时替换
            text=True,  # 输出的内容是字符串，不是二级制字节
            encoding="utf-8",  # 指定中文字符串编码
            bufsize=1  # 按行缓存，只要缓冲区一行满了就输出，模拟实时效果
        )

        # 3.打印子进程日志信息
        for line in proc.stdout:
            self.logger.info(f"执行mineru产生的日志: {line}")

        # 4.等待子进程执行完成
        processed_code = proc.wait()

        process_end_time = time.time()

        if processed_code == 0:
            self.logger.info(
                f"执行mineru解析PDF文件成功{import_file_path.name}，耗时：{process_end_time - process_start_time:.2f}秒")
        else:
            self.logger.error(f"执行mineru解析PDF文件失败{import_file_path.name}，错误码processed_code：{processed_code}")

        # 返回状态码
        return processed_code

    def _execute_mineru_online(
            self,
            import_file_path: Path,
            file_dir_path: Path,
            model_version: str = "vlm",
            poll_interval: float = 5.0,
            timeout: float = 1800.0,
    ) -> int:
        """调用 MinerU 官方在线 API 解析本地文件。

        该方法与本地 ``_execute_mineru`` 相互独立，不会启动本地
        MinerU 模型。它会申请上传地址、上传文件、轮询解析结果，
        然后下载并解压结果。

        需要在环境变量中配置：
            MINERU_API_TOKEN: MinerU API 管理页面创建的 Token。
            MINERU_BASE_URL: API 根地址，默认 https://mineru.net/api/v4。

        Args:
            import_file_path: 待解析的本地文件。
            file_dir_path: 解析结果的输出根目录。
            model_version: 线上模型，可选 ``vlm`` 或 ``pipeline``。
            poll_interval: 轮询任务状态的间隔秒数。
            timeout: 整个线上解析的最长等待秒数。

        Returns:
            0 表示解析成功。生成的 Markdown 会放到与本地
            MinerU 相同的目录结构中，因此可继续使用 ``_get_md_paths``。

        Raises:
            PdfConversionError: 配置缺失、API 请求失败、解析失败或超时。
        """
        token = os.getenv("MINERU_API_TOKEN", "").strip()
        base_url = os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4").rstrip("/")

        if not token:
            raise PdfConversionError("MINERU_API_TOKEN 未配置", self.name)
        if model_version not in {"vlm", "pipeline"}:
            raise PdfConversionError(
                f"不支持的 MinerU 线上模型: {model_version}",
                self.name,
            )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        file_dir_path.mkdir(parents=True, exist_ok=True)
        temp_zip_path = None

        def read_api_data(response: requests.Response, step_name: str) -> dict:
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(
                    f"{step_name}失败: code={payload.get('code')}, "
                    f"msg={payload.get('msg', '未知错误')}"
                )
            return payload.get("data") or {}

        try:
            self.log_step("online-step1", "申请 MinerU 线上文件上传地址")
            create_response = requests.post(
                f"{base_url}/file-urls/batch",
                headers=headers,
                json={
                    "files": [{"name": import_file_path.name}],
                    "model_version": model_version,
                },
                timeout=(10, 60),
            )
            create_data = read_api_data(create_response, "申请上传地址")
            batch_id = create_data.get("batch_id")
            upload_urls = create_data.get("file_urls") or []
            if not batch_id or not upload_urls:
                raise RuntimeError("MinerU 未返回 batch_id 或文件上传地址")

            self.log_step("online-step2", "上传文件到 MinerU 在线服务")
            with import_file_path.open("rb") as file_obj:
                upload_response = requests.put(
                    upload_urls[0],
                    data=file_obj,
                    timeout=(10, 600),
                )
            upload_response.raise_for_status()

            self.log_step("online-step3", f"轮询 MinerU 解析任务: {batch_id}")
            deadline = time.monotonic() + timeout
            result_zip_url = None
            last_state = None

            while time.monotonic() < deadline:
                result_response = requests.get(
                    f"{base_url}/extract-results/batch/{batch_id}",
                    headers=headers,
                    timeout=(10, 60),
                )
                result_data = read_api_data(result_response, "查询解析结果")
                extract_results = result_data.get("extract_result") or []
                current_result = next(
                    (
                        item for item in extract_results
                        if item.get("file_name") == import_file_path.name
                    ),
                    extract_results[0] if extract_results else None,
                )

                if current_result:
                    state = current_result.get("state", "unknown")
                    if state != last_state:
                        self.logger.info(f"MinerU 在线解析状态: {state}")
                        last_state = state

                    if state == "done":
                        result_zip_url = current_result.get("full_zip_url")
                        if not result_zip_url:
                            raise RuntimeError("解析已完成，但未返回 full_zip_url")
                        break
                    if state == "failed":
                        raise RuntimeError(
                            f"MinerU 在线解析失败: "
                            f"{current_result.get('err_msg', '未知错误')}"
                        )

                time.sleep(poll_interval)

            if not result_zip_url:
                raise TimeoutError(
                    f"MinerU 在线解析超时，已等待 {timeout:.0f} 秒，"
                    f"batch_id={batch_id}"
                )

            self.log_step("online-step4", "下载并解压 MinerU 解析结果")
            download_response = requests.get(
                result_zip_url,
                stream=True,
                timeout=(10, 600),
            )
            download_response.raise_for_status()

            with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".zip",
                    dir=file_dir_path,
                    delete=False,
            ) as temp_zip:
                temp_zip_path = Path(temp_zip.name)
                for chunk in download_response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temp_zip.write(chunk)

            target_dir = file_dir_path / import_file_path.stem / "auto"
            target_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(temp_zip_path) as zip_file:
                markdown_members = []
                target_root = target_dir.resolve()
                for zip_info in zip_file.infolist():
                    member_path = Path(zip_info.filename)
                    destination = (target_dir / member_path).resolve()
                    if destination != target_root and target_root not in destination.parents:
                        raise RuntimeError(
                            f"MinerU 结果压缩包包含非法路径: {zip_info.filename}"
                        )
                    if not zip_info.is_dir() and member_path.suffix.lower() == ".md":
                        markdown_members.append(member_path)
                zip_file.extractall(target_dir)

            if not markdown_members:
                raise RuntimeError("MinerU 结果压缩包中没有 Markdown 文件")

            markdown_member = next(
                (path for path in markdown_members if path.name == "full.md"),
                markdown_members[0],
            )
            source_markdown = target_dir / markdown_member
            expected_markdown = target_dir / f"{import_file_path.stem}.md"

            # 如果 ZIP 外面还套了一层目录，把该目录的内容合并到 auto，
            # 使 Markdown 中的图片相对路径仍然有效。
            if source_markdown.parent != target_dir:
                shutil.copytree(source_markdown.parent, target_dir, dirs_exist_ok=True)
                source_markdown = target_dir / source_markdown.name

            if source_markdown != expected_markdown:
                source_markdown.replace(expected_markdown)

            self.logger.info(
                f"MinerU 在线解析成功: {import_file_path.name}, "
                f"Markdown={expected_markdown}"
            )
            return 0
        except (requests.RequestException, RuntimeError, TimeoutError,
                ValueError, zipfile.BadZipFile) as exc:
            raise PdfConversionError(
                f"MinerU 在线解析失败: {exc}",
                self.name,
            ) from exc
        finally:
            if temp_zip_path and temp_zip_path.exists():
                temp_zip_path.unlink()

    def _get_md_paths(self, import_file_path: Path, file_dir_path: Path) -> str:
        """
        获取PDF解析成MD后的MD文件的实际路径
        :param import_file_path: 被解析的PDF文件路径，全路径(路径+文件名)
        :param file_dir_path: 指定的Pdf解析后的MD存储路径
        :return: 实际的MD文件全路径(路径+文件名)
        """
        # pdf文件的名称，不含扩展名和路径
        file_name = import_file_path.stem
        # /Users/mazhiwei/Downloads/项目-智库/shopkeeper_brain_02_终结版/shopkeeper_brain/knowledge/processor/import_process/temp_dir/hak180产品安全手册/auto/hak180产品安全手册.md
        md_path = file_dir_path / file_name / "auto" / f"{file_name}.md"

        # MinerU命令返回成功后，还需要确认目标Markdown文件确实已经生成。
        if not md_path.is_file():
            raise FileProcessingError(
                f"MinerU转换完成，但没有找到Markdown文件：{md_path}",
                self.name
            )

        return str(md_path)


# *******************************************************************#
#                  测试                                              #
# *******************************************************************#

if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = PdfToMdNode()
    pdf_to_md_node_init_state = {
        "import_file_path": r"/Users/mazhiwei/Downloads/项目-智库/shopkeeper_brain_02_终结版/shopkeeper_brain/knowledge/processor/import_process/temp_dir/万用表RS-12的使用.pdf",
        "import_file_path": r"/Users/mazhiwei/Downloads/项目-智库/shopkeeper_brain_02_终结版/shopkeeper_brain/knowledge/processor/import_process/temp_dir/万用表RS-12的使用.pdf",
        "file_dir": r"/Users/mazhiwei/Downloads/项目-智库/shopkeeper_brain_02_终结版/shopkeeper_brain/knowledge/processor/import_process/temp_dir2"
    }
    processed_result = pdf_to_md_node.process(pdf_to_md_node_init_state)
    print(json.dumps(processed_result, indent=4, ensure_ascii=False))


"""
2026-05-31 22:07:08 - import.pdf_to_md_node - INFO - 执行mineru解析PDF文件成功万用表RS-12的使用.pdf，耗时：112.43秒
{
    "import_file_path": "D:\\workspace\\python\\PythonProject\\shopkeeper_brain\\knowledge\\processor\\import_process\\temp_dir\\万用表RS-12的使用.pdf",
    "file_dir": "D:\\workspace\\python\\PythonProject\\shopkeeper_brain\\knowledge\\processor\\import_process\\temp_dir",
    "md_path": "D:\\workspace\\python\\PythonProject\\shopkeeper_brain\\knowledge\\processor\\import_process\\temp_dir\\万用表RS-12的使用\\auto\\万用表RS-12的使用.md"
}
"""
