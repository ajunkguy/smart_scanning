# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:14:09 2019

@author: 90415
"""
import random
import numpy as np
import numba
import copy
import math


'''
模拟池参数
ipnum,dhcp_num,device_num,device_class_num
'''
ac = 50 #加速模拟倍数
ipnum = 32*256
dhcp_num =10
one_time_cost_hour = 4.55
#one_time_cost_hour2 = 4.55
test_time_range = int(24*60*one_time_cost_hour/24/ac)#扫描一遍所需花费的时间
device_num_ratio = 0.8
device_num =int(ipnum*device_num_ratio)
device_class_num = 20
on_rate = 0.99

#random.seed( 1 )
'''
全局变量区
'''

world_time = 0

#设备型号的随机参数,固定IP
seed_table3 = np.zeros(device_class_num)
for i in range(device_class_num):
    seed_table3[i] = random.randint(0,60*24-1)


#设备型号的随机参数,全部动态IP,意义为平均租期
seed_table = np.zeros(device_class_num)
for i in range(device_class_num):
    seed_table[i] = random.randint(1,int(60*24*30/ac))
print('seed_table',seed_table)


'''
#最大租期的随即参数
seed_table2= np.zeros(dhcp_num)
for i in range(dhcp_num):
    seed_table2[i] = random.randint(1,6)
'''

class device():
    # 定义构造方法
    def __init__(self, class_in, address):  #__init__() 是类的初始化方法；它在类的实例化操作后 会自动调用，不需要手动调用；
        # 设置属性
        self.class_id = class_in#设备型号，fingerprint结果
        self.ipa = address
        self.lease_time = 0
        self.dhcp_pool_num = -1
        self.deleted = False
        #self.class_e = np.zeros()
    
    def fresh(self):
        self.deleted = False
        self.lease_time = 0
    
    def if_on(self):
        return not self.if_off()
    
    def if_off(self):
        t_24 = world_time % (24*60)
        c_time = seed_table3[self.class_id]
        if (t_24-c_time)% (24*60) < (24*60)*on_rate:
            return False
        return True
    
    def detectable(self):
        '''
        t_24 = world_time % (24*60)
        c_time = seed_table3[self.class_id]
        if (t_24-c_time)% (24*60) < (24*60)*on_rate:
            return True
        return False
        '''
        return True
  
    '''
    def scene2(self):#每天定时换一次IP地址
        t_24 = world_time % (24*60)
        c_time = seed_table3[self.class_id]
        if 0<= (t_24-c_time) and (t_24-c_time) < ac:
            return True
        
        return False
    '''

    # 判断下一阶段IP地址会不会发生改变
    def change_or_not(self):
        E = seed_table[self.class_id]
        #E = 3600.0
        p = 1.0/E
        if p> random.random():
            return True
        #return self.scene2()
    
    
    def step(self):
        self.lease_time+=1
    
    
class dhcp_pool():
    def __init__(self): 
        self.ip_start = 0
        self.ip_end = 0
        self.ip_free = list(range(self.ip_start,self.ip_end+1))
        self.devices = []

    #给第i个设备换个ip
    def rand_change_ip(self,i_num):
        newip = random.sample(self.ip_free,1)[0]
        self.ip_free.append(self.devices[i_num].ipa)
        self.devices[i_num].ipa = newip
        self.devices[i_num].lease_time=0
        self.ip_free.remove(newip)
        
        return
    
    def delete_one(self,i_num,with_deleted = True):
        self.ip_free.append(self.devices[i_num].ipa)
        #将device的deleted置为True
        if with_deleted:
            self.devices[i_num].deleted =True
        del self.devices[i_num]
    
    def new_one(self,newdevice):
        if len(self.ip_free)>1:
            newip = random.sample(self.ip_free,1)[0]
            newdevice.ipa = newip
            newdevice.lease_time=0
            self.ip_free.remove(newip)
            self.devices.append(newdevice)
            return True
        else:
            return False


class Dead_Pool():
    def __init__(self): 
        self.devices = []
        
    
    
    
class IP_Pool():
    def __init__(self, ipnum,dhcp_num,device_num,device_class_num =10): 
        # 设置属性
        self.ipnum = ipnum
        self.device_num = device_num
        self.device_class_num = device_class_num
        self.device_occurrence_p = np.zeros(device_class_num)
        self.devicelist = []
        self.ip_search_dict = {}
        self.dead_p = Dead_Pool()
        self.dhcp_num = dhcp_num
        self.renew_devices()
        self.renew_dhcp_pool()
        self.renew_search_dict()
        
    def renew_search_dict(self):
        self.ip_search_dict = {}
        for i in range(len(self.devicelist)):
            ipa = self.devicelist[i].ipa
            self.ip_search_dict[ipa] = i   
        #print(self.ip_search_dict)
        return
        
    #随机device_num个设备，共device_class_num种
    def rand_device_classes(self):
        results = np.zeros(self.device_class_num)
        cutpoints = random.sample(range(self.device_num),self.device_class_num-1)
        cutpoints.append(self.device_num)
        cutpoints.sort()
        for i in range(len(results)):
            if i>0:
                results[i] = cutpoints[i]-cutpoints[i-1]
        results[0] = cutpoints[0]
        
        #将归一化概率写入device_occurrence_p
        for i in range(len(results)):
            self.device_occurrence_p[i] = results[i]/float(self.device_num)
            
        return results

    #初始化设备参数
    def renew_devices(self):
        self.devicelist = []
        #这部分改到renew_dhcp_pool函数里面写
        '''
        rand_classes = self.rand_device_classes()

        
        #获取初始化设备
        for i in range(len(rand_classes)):
            for j in range(int(rand_classes[i])):
                newd = device(i,-1)
                self.devicelist.append(newd)
                
                
        #获取初始化ip地址
        points = random.sample(range(self.ipnum),self.device_num)
        for i in range(len(self.devicelist)):
            self.devicelist[i].ipa = points[i]
            #print(self.devicelist[i].ipa)
        '''
        return
    
    #初始化dhcp池子的分布
    def build_dhcp_device_matricx(self,dhcp_cutpoints):
        rand_classes = self.rand_device_classes()
        rand_classes = rand_classes/sum(rand_classes)
        #print(rand_classes)

        out_mat = np.zeros((len(dhcp_cutpoints),len(rand_classes)))
        
        for i in range(len(dhcp_cutpoints)):
            rand_classes2 = self.rand_device_classes()
            rand_classes2 = rand_classes2/sum(rand_classes2)
            #print(rand_classes2)
            out_mat[i,:] = rand_classes * rand_classes2
            out_mat[i,:] = out_mat[i,:]/sum(out_mat[i,:])
            
        #print(out_mat)
        
        #os.exit()
        return  out_mat
    
    #初始化dhcp池子的分布
    #@numba.jit
    def renew_dhcp_pool(self):
        C_number = int(self.ipnum/256)
        range1 = range(1,C_number)
        cutpoints = random.sample(range1,self.dhcp_num-1)
        cutpoints.append(C_number)
        cutpoints.sort()
        
        print('C_number',C_number)
        print('cutpoints',cutpoints)
        self.dhcp_c_list = cutpoints
        
        
        #计算出每个每个dhcp池子里面应有的设备数目，并添加设备
        #rand_classes = self.rand_device_classes()#[设备1的数目，设备二的数目]
        dhcp_mat = self.build_dhcp_device_matricx(cutpoints)
        print('dhcp_mat',dhcp_mat)
        
        
        self.dhcp_pool_list = []
        for i in range(self.dhcp_num):
            dhcpi = dhcp_pool()
            if i == 0:
                dhcpi.ip_start = int(0*256)
                dhcpi.ip_end = int(cutpoints[0]*256)-1
            else:
                dhcpi.ip_start = int(cutpoints[i-1]*256)
                dhcpi.ip_end =int(cutpoints[i]*256)-1
            
            freelist = list(range(dhcpi.ip_start,dhcpi.ip_end+1))
            dhcpi.ip_free = freelist
            
            #添加设备
            for i_device_type in range(len(dhcp_mat[i,:])):
                device_add_num = round(dhcp_mat[i,i_device_type] * device_num_ratio * (dhcpi.ip_end-dhcpi.ip_start))
                #生成设备
                #print(i,i_device_type,device_add_num)
                for i_device_num in range(int(device_add_num)):
                    newdevice = device(i_device_type,-1)
                    newdevice.dhcp_pool_num=i
                    if dhcpi.new_one(newdevice):
                        self.devicelist.append(newdevice)

            '''
            #偷懒写法
            for i_d_num in range(len(self.devicelist)):
                if self.devicelist[i_d_num].ipa>=dhcpi.ip_start and self.devicelist[i_d_num].ipa<=dhcpi.ip_end:
                    self.devicelist[i_d_num].dhcp_pool_num=i
                    dhcpi.devices.append(self.devicelist[i_d_num])
                    #print(self.devicelist[i_d_num].ipa)
                    dhcpi.ip_free.remove(self.devicelist[i_d_num].ipa)
                    #print(i)
            '''
            
            self.dhcp_pool_list.append(dhcpi)
            print('lend',len(self.dhcp_pool_list[i].devices))
            


    #测试函数，没什么用
    def test(self):
        for i in range(len(self.dhcp_pool_list)): 
            for j in range(len(self.dhcp_pool_list[i].devices)):
                print(self.dhcp_pool_list[i].devices[j].class_id)
                
        self.devicelist[0].class_id = 100
        
        for i in range(len(self.dhcp_pool_list)): 
            for j in range(len(self.dhcp_pool_list[i].devices)):
                print(self.dhcp_pool_list[i].devices[j].class_id)
    
    
    #模拟真实世界的变化,可以当成每1min变化一次
    #@numba.jit
    def world_step(self):
        global world_time
        world_time += ac
        
        for i in range(len(self.dhcp_pool_list)): 
            
            '''
            max_lease_p =  seed_table2[i]
            if max_lease_p <=3:
                max_lease_p = max_lease_p*60*24
            else:
                max_lease_p = 9999999999#一个超大数字
            '''
            delnumber = 0
            for j in range(len(self.dhcp_pool_list[i].devices)):
                nowdevice = self.dhcp_pool_list[i].devices[j-delnumber]
                nowdevice.step()
                
                #最大租期是否到了
                max_lease_flag = False
                #if nowdevice.lease_time>= max_lease_p:
                #    max_lease_flag =True
                    
                #随机动态变化地址
                
                if nowdevice.change_or_not() == True or max_lease_flag == True:
                    self.dhcp_pool_list[i].rand_change_ip(j-delnumber)
                '''
                    
                #如果设备关了，那么就放进亡语池子
                #设备亡语：在某个判断为on的时刻重新连接进来。。。。
                if nowdevice.if_off():
                    self.dead_p.devices.append(copy.deepcopy(nowdevice))
                    self.dhcp_pool_list[i].delete_one(j-delnumber,True)
                    delnumber += 1
                '''
        
        #触发设备亡语
        self.device_dead_on()
        #self.devicein()
        #self.deviceoff()
        
        
        #偷懒写法,辅助之前的删除操作
        self.delete_null()
        self.renew_search_dict()
        
        return 

    def rand_device_type(self):
        a = random.random()
        for i  in range(len(self.device_occurrence_p)):
            a -= self.device_occurrence_p[i]
            if a<=0:
                return i
    
    def device_dead_on(self):
        delnumber = 0
        for j in range(len(self.dead_p.devices)): 
            i_dev =  self.dead_p.devices[j-delnumber]
            if i_dev.if_on():
                i_dev.fresh()
                pool_num = i_dev.dhcp_pool_num
                if self.dhcp_pool_list[pool_num].new_one(i_dev):
                    self.devicelist.append(i_dev)
                    del self.dead_p.devices[j-delnumber]
                    delnumber += 1
        return
    
    #以恒定的速率进入新的设备
    def devicein(self):
        average_time = 60/ac
        p_value = 1.0/average_time/device_num
        for i in range(device_num):
            if random.random()<p_value:
                pool_num = random.randint(0,dhcp_num-1)
                devicetype = self.rand_device_type()
                newdevice = device(devicetype,-1)
                #print(newdevice.class_id)
                newdevice.dhcp_pool_num = pool_num
                
                if self.dhcp_pool_list[pool_num].new_one(newdevice):
                    self.devicelist.append(newdevice)
                #print(newdevice.ipa)
                #print(self.devicelist[len(self.devicelist)-1].ipa,'aaa')
        return 
    
    def delete_null(self):
        delnumber = 0
        #将所有标为null的删除
        #print(len(self.devicelist))
        for i in range(len(self.devicelist)):
            device = self.devicelist[i-delnumber]
            if device.deleted:
                #print('deleted')
                del self.devicelist[i-delnumber]
                
                delnumber += 1
        
    #每个设备以恒定的概率退出
    def deviceoff(self):
        average_time = 60/ac
        p_value = 1.0/average_time/device_num
        #print(p_value)
        
        for i in range(len(self.dhcp_pool_list)): 
            delnumber = 0
            for j in range(len(self.dhcp_pool_list[i].devices)):   
                #一定概率删除该设备
                if random.random()<p_value:
                    #print('delete one...')
                    #print(i,j,len(self.dhcp_pool_list[i].devices),delnumber)
                    #device = self.dhcp_pool_list[i].devices[j-delnumber]
                    #断掉池子的连接,并把删除的设备置为NULL
                    #print(len(self.devicelist),len(self.dhcp_pool_list[i].devices))
                    self.dhcp_pool_list[i].delete_one(j-delnumber,True)
                    #print(len(self.devicelist),len(self.dhcp_pool_list[i].devices))
                    #os.exit()
                    #print(self.devicelist[i])
                    delnumber += 1
                
        #self.delete_null()
        #print(len(self.devicelist))
        return 

    def test_one_ip(self,ipa):
        if ipa in self.ip_search_dict.keys():
            i_num = self.ip_search_dict[ipa]
            #print(self.devicelist[i_num].detectable())
            if self.devicelist[i_num].detectable():
                return self.devicelist[i_num].class_id
            else:
                return device_class_num
        else:
            return device_class_num


#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Deep_QN(nn.Module):
    def __init__(self):
        super(Deep_QN, self).__init__()
        #self.conv1 = nn.Conv1d(3, 20, 5, 1)
        self.embedding1 = nn.Embedding(ipnum+1, 5)
        self.embedding2 = nn.Embedding(device_class_num+1, 5)

        self.ftest = nn.Linear(10,30)
        
        self.fc1 = nn.Linear(11, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10,1)

    def forward(self, xyz):
        ipa= xyz[0]
        type1 = xyz[1]
        time = xyz[2]
        #print(ipa,type1,time)
        
        m1 = self.embedding1(ipa)
        m2 = self.embedding2(type1)
        
        m12 = torch.cat((m1,m2),2)
        m12 = m12.view(m12.shape[0],-1)
        
       # print(m12.shape)
        m123 = torch.cat((m12,time),1)
        #print(m123.shape)
        
        #x = F.relu(self.ftest(m2))
        x = torch.tanh(self.fc1(m123))
        #print(x.shape)
        x = torch.tanh(self.fc2(x))
        #print(x.shape)
        x = torch.tanh(self.fc3(x))
        #print(x.shape)
        x = torch.tanh(self.fc4(x))

        return torch.sigmoid(x)
        #return F.log_softmax(x, dim=1)


#print (len(dataset.trainset))
net = Deep_QN()


#%%
class Experience():
    # 定义构造方法
    def __init__(self):  #__init__() 是类的初始化方法；它在类的实例化操作后 会自动调用，不需要手动调用；
        self.exp = []
    
    def add_exp(self,object1):
        self.exp.append(object1)
        
        
    #需要正负平衡！！！！
    def replay(self,batch_size):
        tensor_ipa = torch.LongTensor(batch_size, 1).zero_()
        tensor_type1 = torch.LongTensor(batch_size, 1).zero_()
        tensor_time = torch.FloatTensor(batch_size, 1).zero_()
        tensor_results = torch.FloatTensor(batch_size).zero_()
        
        for i in range(batch_size):
            index = random.randint(0,len(self.exp)-1)
            data = self.exp[index]
            #print(data[0])
            tensor_ipa[i,0] = data[0]
            tensor_type1[i,0] = data[1]
            tensor_time[i,0] = data[2]
            tensor_results[i] = data[3]
        
        return [tensor_ipa,tensor_type1,tensor_time,tensor_results]

class NoExperience():
    def __init__(self):
        self.exp = []
        #self.poolnum =1000 

    def add_exp(self,object1):
        self.exp.append(object1)
        #print(object1)

    def replay(self,batch_size):
        allnumber = len(self.exp)
        tensor_ipa = torch.LongTensor(allnumber, 1).zero_()
        tensor_type1 = torch.LongTensor(allnumber, 1).zero_()
        tensor_time = torch.FloatTensor(allnumber, 1).zero_()
        tensor_results = torch.FloatTensor(allnumber).zero_()

        for i in range(allnumber):
            data = self.exp[i]
            tensor_ipa[i,0] = data[0]
            tensor_type1[i,0] = data[1]
            tensor_time[i,0] = data[2]
            tensor_results[i] = data[3]
        
        self.exp = []
        
        return [tensor_ipa,tensor_type1,tensor_time,tensor_results]

experience = NoExperience()        


#%%
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import time

def exp_func(x,lambda1):
    return 1 - np.exp(-x*lambda1)

'''
def func(x, a, b, c):
    return a * np.exp(-b * x) + c


xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
ydata = y + 0.2 * np.random.normal(size=len(xdata))
plt.plot(xdata,ydata,'b-')
popt, pcov = curve_fit(func, xdata, ydata)
#popt数组中，三个值分别是待求参数a,b,c
y2 = [func(i, popt[0],popt[1],popt[2]) for i in xdata]
plt.plot(xdata,y2,'r--')
print (popt)
'''

def draw(xdata,ydata):
    #plt.plot(xdata,ydata,'b-')
    plt.scatter(xdata, ydata, marker = 'x',color = 'blue', s = 40)
    popt, pcov = curve_fit(exp_func, xdata, ydata,p0=0.001,bounds =(0,0.05))
    xdata = np.linspace(0, 1000, 50)
    y2 = [exp_func(i, popt[0]) for i in xdata]
    plt.plot(xdata,y2,'r--')
    plt.show()
    print(popt)
    

#draw([0,100,200,300,400,500],[0,1,0,0,0,0])

#%%
def help_get_lambda(starttime,endtime):
    result_y = np.zeros(24)
    
    now = starttime
    down = int(starttime / 60) * 60
    up = math.ceil(starttime / 60) * 60
    while(True):
        if endtime<up:
            result_y[int(down/60)%24] += (endtime-now)/(endtime-starttime)
            #sum_lt += ((endtime-now)/60) * lambdalist[int(down/60)%24]
            break
        #sum_lt += ((up-now)/60) * lambdalist[int(down/60)%24]
        result_y[int(down/60)%24] += (up-now)/(endtime-starttime+0.01)
        now = up
        down = up
        up+=60
    
    return result_y
    
#print(help_get_lambda(1*60,7.5*60))
    
def get_lambda(scan_list,isc,upper_c = True):
    #如果数据太少，，直接返回默认值
    #if len(scan_list)<=5:
    #   return tuple(list(np.ones(24)/10))
    
    global scan_weight_class_y7
    global scan_weight_class_n7
    
    for i_scan in scan_list:
        (t1,t2,yon) = i_scan
        if yon == True:
            scan_weight_class_y7[isc] = help_get_lambda(t1,t2) + scan_weight_class_y7[isc]
        else:
            #scan_weight_class_n7[isc] = help_get_lambda(t1,t2) + scan_weight_class_n7[isc]
            scan_weight_class_n7[isc]=help_get_lambda(t1,t2)/max(help_get_lambda(t1,t2))+scan_weight_class_n7[isc]
    
    ''' 
    print(scan_list)
    print('scan_weight_class_y7',scan_weight_class_y7)
    print('scan_weight_class_n7',scan_weight_class_n7)
    print(scan_weight_class_y7[isc]/(scan_weight_class_y7[isc]+scan_weight_class_n7[isc]+1))
    '''
    
    upper_confidence = 1/(scan_weight_class_y7[isc]+scan_weight_class_n7[isc]+1)
    #print(upper_confidence)
    #os.exit()
    if upper_c:
        return upper_confidence+(scan_weight_class_y7[isc]/(scan_weight_class_y7[isc]+scan_weight_class_n7[isc]+0.001))
    else:
        return (scan_weight_class_y7[isc]/(scan_weight_class_y7[isc]+scan_weight_class_n7[isc]+0.001))

def cacul_sum_p(starttime,endtime,lambdalist):
    now = starttime
    down = int(starttime / 60) * 60
    up = math.ceil(starttime / 60) * 60
    not_change = 1
    while(True):
        if endtime<up:
            not_change *= (1 - min(1,((endtime-now)/60) * lambdalist[int(down/60)%24]))
            #sum_lt += ((endtime-now)/60) * lambdalist[int(down/60)%24]
            break
        
        not_change *= (1 - min(1,((up-now)/60) * lambdalist[int(down/60)%24]))
        #sum_lt += ((up-now)/60) * lambdalist[int(down/60)%24]
        now = up
        down = up
        up+=60
    return 1 - not_change
    
def cacul_sum_p_no_device(starttime,endtime,lambdalist,big_l_list,device_mat7,testip):
    #average_big_l_list = np.array(big_l_list).sum(axis=0)/len(big_l_list[0])
    average_l_list = np.zeros(24)
    for i_type in range(device_class_num):
        average_l_list += big_l_list[i_type]*device_mat7[int(testip/256),i_type]
    device_num_onepool = 0
    for i_type in range(device_class_num):
        device_num_onepool += device_mat7[int(testip/256),i_type]
    average_l_list = average_l_list/device_num_onepool
    #print(average_l_list)

    now = starttime
    down = int(starttime / 60) * 60
    up = math.ceil(starttime / 60) * 60
    not_change = 1
    while(True):
        if endtime<up:
            not_change = not_change*(1 - min(1,((endtime-now)/60) * lambdalist[int(down/60)%24])) + (1-not_change)*((endtime-now)/60)*average_l_list[int(down/60)%24]
            #sum_lt += ((endtime-now)/60) * lambdalist[int(down/60)%24]
            break
        
        not_change = not_change*(1 - min(1,((up-now)/60) * lambdalist[int(down/60)%24])) + (1-not_change)*((up-now)/60)*average_l_list[int(down/60)%24]
        
        #sum_lt += ((up-now)/60) * lambdalist[int(down/60)%24]
        now = up
        down = up
        up+=60
    return 1 - not_change
    
#print(cacul_sum_p(60,120,np.ones(24)/2))


def back_to_lamda(lambdalist):
    size = 24
    A = np.zeros((size,size))
    for i in range(size):
        starttime =  (i+0.5) - one_time_cost_hour
        endtime = (i+0.5) + one_time_cost_hour
        for j in range(int(starttime),int(endtime)):
            A[i,j%size] = 1 * (one_time_cost_hour-abs(j-i+0.5))/one_time_cost_hour
            #A[i,j%size] = (1/one_time_cost_hour)  * (one_time_cost_hour-abs(j-i))/one_time_cost_hour
        #A[i,:] = A[i,:]/sum(A[i,:])
    
    #print(A)
    #print(np.linalg.inv(A))
    #result = np.dot(A.T,A)
    result = np.linalg.inv(A)
    #result = np.dot(result,A.T)
    result = np.dot(result,lambdalist)
    
    
    for ir in range(len(result)):
        if result[ir] <=0:
            result[ir] = 0
    
    if max(result)>1:
        result = result / max(result)
    
    return result

testnp = np.ones(24)
#testnp[0:5] = 0.2
#print(testnp)
#print(back_to_lamda(testnp))


#%%
#@numba.jit

from torch.autograd import Variable
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)#, weight_decay=0.0001)
loss_func = nn.MSELoss()



def start_test():
    
    global one_time_cost_hour
    global one_time_cost_hour2
    global test_time_range
    

    test_time_range = int(24*60*one_time_cost_hour/24/ac)#扫描一遍所需花费的时间

    ip_pool_1 = IP_Pool(ipnum,dhcp_num,device_num,device_class_num)
    '''ipnum,dhcp_num,device_num,device_class_num'''
    

    '''
    初始化打乱
    '''
    print('Initialization start...')
    for i in range(int(60*24*30*1/ac)):
        ip_pool_1.world_step()
    print('Initialization finish...')
    
    
    '''
    以下为测试方案一的参数，即扫描频率为常数
    '''
    frequency_times1 = ipnum/test_time_range
    frequency_sum1 = 0
    ip_memory = np.zeros(ipnum)
    for testip in range(ipnum):
        ip_memory[testip] = ip_pool_1.test_one_ip(testip)
    search_index1 = 0

    
    '''
    以下为测试方案二的参数，即扫描频率为常数
    '''
    frequency_times2 = ipnum/test_time_range
    frequency_sum2 = 0
    ip_memory2 = np.zeros(ipnum)
    for testip in range(ipnum):
        ip_memory2[testip] = ip_pool_1.test_one_ip(testip)
    search_index2 = 0
    
    '''
    以下为测试方案5的参数
    '''
    ip_memory5 = np.zeros(ipnum).astype('int32')
    for testip in range(ipnum):
        ip_memory5[testip] = ip_pool_1.test_one_ip(testip)
    frequency_times5 = ipnum/test_time_range
    frequency_sum5 = 0
    
    test_time_map5  = np.zeros(ipnum)
    
    
    
    '''
    以下为测试方案7的参数
    '''
    
    ip_memory7 = np.zeros(ipnum).astype('int32')
    for testip in range(ipnum):
        ip_memory7[testip] = ip_pool_1.test_one_ip(testip)
        
    frequency_times7 = ipnum/test_time_range
    frequency_sum7 = 0
    
    last_world_time7  = np.zeros(ipnum)
    for testip in range(ipnum):
        last_world_time7[testip] = world_time-1
        
    #记录所有探测的数据
    scan_device_list7 = []
    for isc in range(device_class_num+1):
        scan_device_list7.append([])
    
    #记录扫描贡献
    global scan_weight_class_y7
    global scan_weight_class_n7
    scan_weight_class_y7 = np.zeros((device_class_num+1,24))
    scan_weight_class_n7 = np.zeros((device_class_num+1,24))
    
    #记录扫描分布
    ip_pool_min_range = 256
    pool_num = int(ipnum / ip_pool_min_range)
    device_mat7 = np.zeros((pool_num,device_class_num+1))
    
    '''
    以下为统计判断参数
    '''    
    value1_sumall = 0
    value2_sumall = 0
    value3_sumall = 0
    value4_sumall = 0
    value5_sumall = 0
    value6_sumall = 0
    value7_sumall = 0
    
    value1_sumall_mem = 0
    value2_sumall_mem = 0
    value3_sumall_mem = 0
    value4_sumall_mem = 0
    value5_sumall_mem = 0
    value6_sumall_mem = 0
    value7_sumall_mem = 0
    
    cishu1 = 0
    cishu2 = 0
    cishu3 = 0
    cishu4 = 0
    cishu5 = 0
    cishu6 = 0
    cishu7 = 0
    
    white_cishu1 = 0
    white_cishu2 = 0
    white_cishu5 = 0
    white_cishu7 = 0
    
    '''
    以下为聚类用参数
    '''
    #转移概率矩阵大小为设备数目*ip池粒度
    ip_pool_min_range = 256
    pool_num = int(ipnum / ip_pool_min_range)
    d_change_mat = np.zeros((pool_num,device_class_num))
    
    

    
    
    for i in range(int(60*24*30*12*10/ac)):
        ip_pool_1.world_step()
        
        #进行测试，测试方案一
        frequency_sum1 += frequency_times1
        while(frequency_sum1>1):
            frequency_sum1 -= 1
            '''
            target = search_index1
            search_index1 +=1
            search_index1 = search_index1%ipnum
            '''
            target = random.randint(0,ipnum-1)
            
            result = ip_pool_1.test_one_ip(target)

            cishu1 += 1
            if result != ip_memory[target]:
                value1_sumall+=1
            
            if ip_memory[target] == device_class_num:
                white_cishu1 += 1
            
            ip_memory[target] = result

            
            
        #测试方案二，顺序扫描
        frequency_sum2 += frequency_times2
        while(frequency_sum2>1):
            frequency_sum2 -= 1
            target = search_index2
            search_index2 +=1
            search_index2 = search_index2%ipnum
            
            
            result = ip_pool_1.test_one_ip(target)

            cishu2 += 1
            if result != ip_memory2[target]:
                value2_sumall+=1
            
            if ip_memory2[target] == device_class_num:
                white_cishu2 += 1
            
            ip_memory2[target] = result

        
        '''测试方案三'''
        '''
        DQN in v3
        '''
        
        #方法四
        '''
        best lamda
        '''
        
        #方法五
        #更新预判概率
        '''
        predict_outmap5 = np.zeros(ipnum)
        test_time_map5 +=1
        for testip in range(ipnum):
            type5 = ip_memory5[testip]
            if int(type5) == device_class_num:
                predict_outmap5[testip] = 0.1#1-math.exp(-0.01*test_time_map5[int(type5)])
                continue
            else:
                t_24 = world_time % (24*60)
                c_time = seed_table3[int(type5)]

                if ((t_24-c_time)% (24*60) < (24*60)*on_rate):#处于开机状态
                    predict_outmap5[testip] = 0
                else:
                    predict_outmap5[testip] = 1
        
        #选取改变概率最高的那个地址进行扫描
        frequency_sum5 += frequency_times5
        while(frequency_sum5>1):
            frequency_sum5 -= 1
            max_index = np.argmax(predict_outmap5)
            if predict_outmap5[max_index]>0.5:
                target = max_index
            else:
                target = np.argmax(test_time_map5)
            result = ip_pool_1.test_one_ip(target)
            
            test_time_map5[target] = 0

            #扫描计数
            cishu5 += 1
            if result != ip_memory5[target]:
                if result == device_class_num:
                    value5_sumall+=1
                else:
                    value5_sumall+=1
                    
                 
            if ip_memory5[target] == device_class_num:
                white_cishu5 += 1
                
            #更新扫描结果
            ip_memory5[target] = result
            predict_outmap5[target] = 0
        '''




        '''测试方案七'''
        device_lambda_list7 = []
        for isc in range(device_class_num+1):
            #device_lambda_list7.append(back_to_lamda(get_lambda(scan_device_list7[isc],isc)))
            device_lambda_list7.append(get_lambda(scan_device_list7[isc],isc))
            scan_device_list7[isc] = []#清空
        #print(scan_weight_class_y7[0],scan_weight_class_n7[0])
        #print('233333')
        #print(device_lambda_list7[0])
        #print(back_to_lamda(device_lambda_list7[0]))
        #print(seed_table3[0],int(seed_table3[0]/60))
        
        predict_outmap7 = np.zeros(ipnum)
        for testip in range(ipnum):
            type7 = ip_memory7[testip]
            #if not type7 == device_class_num:
            predict_outmap7[testip] = cacul_sum_p(last_world_time7[testip],world_time,device_lambda_list7[type7])
            #else: #修正概率
            #    predict_outmap7[testip] = cacul_sum_p_no_device(last_world_time7[testip],world_time,device_lambda_list7[type7],device_lambda_list7[:-1],device_mat7,testip)
                #print(cacul_sum_p(last_world_time7[testip],world_time,device_lambda_list7[type7]),
                 #     cacul_sum_p_no_device(last_world_time7[testip],world_time,device_lambda_list7[type7],device_lambda_list7[:-1],device_mat7,testip))
                #predict_outmap7[testip] = world_time - last_world_time7[target]
            
            
        #根据计算的变化概率，来选择最高概率的IP地址依次扫描
        frequency_sum7 += frequency_times7
        while(frequency_sum7>1):
            frequency_sum7 -= 1        
            target7 = np.argmax(predict_outmap7)
            result7 = ip_pool_1.test_one_ip(target7)

            #扫描计数
            cishu7 += 1
            if result7 != ip_memory7[target7]:
                value7_sumall+=1
            
            if ip_memory7[target7] == device_class_num:
                white_cishu7 += 1
            
            #记录扫描历史
            one_scan = (last_world_time7[target7],world_time,result7 != ip_memory7[target7])
            scan_device_list7[ip_memory7[target7]].append(one_scan)
            
            
            #更新一些扫描结果变量
            ip_memory7[target7] = result7
            last_world_time7[target7] = world_time
            predict_outmap7[target7] = 0
        
        

        '''显示结果'''
        if i%test_time_range == test_time_range-1:  
            tg5 = ((value5_sumall-value5_sumall_mem)-(value1_sumall-value1_sumall_mem)
                  )/(value1_sumall-value1_sumall_mem+0.1)
            tga5 = (value5_sumall-value1_sumall)/(value1_sumall+0.1)
            tg7 = ((value7_sumall-value7_sumall_mem)-(value1_sumall-value1_sumall_mem)
                  )/(value1_sumall-value1_sumall_mem+0.1)
            tga7 = (value7_sumall-value1_sumall)/(value1_sumall+0.1)
            tg2 = ((value2_sumall-value2_sumall_mem)-(value1_sumall-value1_sumall_mem)
                  )/(value1_sumall-value1_sumall_mem+0.1)
            tga2 = (value2_sumall-value1_sumall)/(value1_sumall+0.1)
            tg5 = round(tg5,5)
            tga5 = round(tga5,5)
            tg7 = round(tg7,5)
            tga7 = round(tga7,5)
            print('value123',value1_sumall-value1_sumall_mem,
                  value2_sumall-value2_sumall_mem,
                  value7_sumall-value7_sumall_mem,
                  'part:',tg2,tg7,'all:',tga2,tga7)
   
            value1_sumall_mem = value1_sumall
            value2_sumall_mem = value2_sumall
            #value3_sumall_mem = value3_sumall
            #value4_sumall_mem = value4_sumall
            value5_sumall_mem = value5_sumall
            value7_sumall_mem = value7_sumall

            print('cishu:',cishu1,cishu2,cishu7)
            print('white_cishu',white_cishu1,white_cishu2,white_cishu7)
            
            cishu1 = 0
            cishu2 = 0
            #cishu3 = 0
            #cishu4 = 0
            cishu5 = 0
            #cishu6 = 0
            cishu7 = 0
            
    
            
            if i/test_time_range>=100:
                return tga7+1
            

'''
写数据
'''
wfile = open('result3.txt', 'w') #清空文件内容再写


result = []
#global device_num
#global device_class_num
for i_time in range(1,21):
    '''
    device_num_ratio = i/10
    device_num =int(ipnum*device_num_ratio)
    device_class_num = 20
    '''

    world_time = 0

    #设备型号的随机参数,固定IP
    seed_table3 = np.zeros(device_class_num)
    for i in range(device_class_num):
        seed_table3[i] = random.randint(0,60*24-1)


    #设备型号的随机参数,全部动态IP,意义为平均租期
    seed_table = np.zeros(device_class_num)
    for i in range(device_class_num):
        seed_table[i] = random.randint(1,int(60*24*30/ac))
    print('seed_table',seed_table)

    per_s = i_time/10

    one_time_cost_hour = ipnum / (per_s*60*60)
    print('one_time_cost_hour',one_time_cost_hour)
    test_time_range = int(24*60*one_time_cost_hour/24/ac)#扫描一遍所需花费的时间

    sumtg = 0
    for i_j in range(10):
        sumtg += start_test() 
    print(sumtg/10)
    result.append(sumtg/10)

    wfile.write(str(per_s) +' ' + str(sumtg/10))
    wfile.write('\n')


wfile.close()
print('result',result)


'''
sumtg = 0
for i in range(10):
    sumtg += start_test() 
print(sumtg/10)
'''

#start_test() 

