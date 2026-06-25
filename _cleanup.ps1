# Aegis 清理脚本 - 删除所有临时备份文件
# 右键以管理员身份运行 PowerShell，然后执行此脚本

Get-ChildItem -Path "D:\Aegis" -Recurse -Filter "*.tmp.*" | Remove-Item -Force -Verbose
Remove-Item -Path "D:\Aegis\_cleanup.ps1" -Force -ErrorAction SilentlyContinue
Write-Host "清理完成！"
