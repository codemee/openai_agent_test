from agents import Agent, function_tool, WebSearchTool
import subprocess

@function_tool
def shell_helper(comment: str, shell_command: str) -> str:
    """可以在 Windows 下執行 powershell 指令的工具函式

    Args:
        comment (str): 說明文字
        shell_command (str): 要執行的指令
    """
    print(f'AI 建議執行以下指令：\n\n{comment}\n\n'
          f'```\n{shell_command}\n```\n')
    print('開始執行：\n\n')

    # 啟動子行程
    process = subprocess.Popen(
        ['powershell', '-Command', shell_command],
        shell=True,             # 在 shell 中執行
        stdout=subprocess.PIPE, # 擷取標準輸出
        stderr=subprocess.PIPE, # 擷取錯誤輸出
        text=True               # 以文字形式返回
    )

    result = '執行結果：\n```\n'

    # 即時讀取輸出
    while True:
        output = process.stdout.readline()
        # 如果沒有輸出且行程結束
        if output == '' and process.poll() is not None:
            break
        if output:
            result += output
            print(f"> {output.strip()}")

    # 檢查錯誤輸出
    error = process.stderr.read()
    if error:
        result += f"\n\n錯誤: {error}"
        print(f"錯誤: {error}")

    # 等待行程結束並取得返回碼
    return_code = process.wait()
    result += f"\n```\n命令執行完成，返回碼: {return_code}\n\n"
    print(f"\n\n命令執行完成，返回碼: {return_code}")

    return result
