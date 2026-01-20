from langchain_community.tools import ShellTool

search_tool = ShellTool()

result = search_tool.invoke("ls -a")

print(result)
