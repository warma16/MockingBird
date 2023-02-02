#现在你要帮我写一个扫描内网内所有ip网段的名字
#内网内的ip形如192.168.a.b 0<=a<255 0<=b<=255
#之后给我返回个列表[ip1,1p2]
import os
#from pythonping import ping
class Logger():
    def __init__(self):
        self.f=open("log1.txt","w",encoding="utf-8")
    def close(self):
        self.f.close()
    def log(self,content):
        self.f.write(content)
logger=Logger()
def inner_ip_scanner(logger):
    #现在你要帮我写一个扫描内网内所有ip网段的名字
    #内网内的ip形如192.168.a.b 0<=a<255 0<=b<=255
    #之后给我返回个列表[ip1,1p2]
    ip_list=[]
    def ip_onsuccess(ip_list):
        print(ip+" is success")
        ip_list.append(ip)
    def ip_onfailed():
        print(ip+" is failed")
    def build_command(ip):
        return "ping -n 1 "+str(ip)
    continue_failed_times=0
    for a in range(256):
        for b in range(255):
            print(str(a*256+(b+1))+"/"+str(256*256))
            logger.log(str(a*256+(b+1))+"/"+str(256*256))
            suffix=str(a)+"."+str(b+1)
            ip="192.168."+suffix
            if os.system(build_command(ip))==0:
                print(ip+" is success")
                logger.log(ip+" is success")
                ip_list.append(ip)
            else:
                ip_onfailed()
                continue_failed_times+=1
                if continue_failed_times>=10:
                    continue_failed_times=0
                    break
                if b+1==1:
                    continue_failed_times=0
                    break
    return ip_list
            


result=inner_ip_scanner(logger)
print(result)
logger.log("\n".join(result))
logger.close()
#os.system("shutdown -s -t 3")
#print(inner_ip_scanner())
