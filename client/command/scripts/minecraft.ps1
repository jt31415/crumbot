$MinecraftAppId = (Get-StartApps -Name "Minecraft Launcher" | Select-Object -First 1).AppID
explorer shell:appsfolder\$MinecraftAppId