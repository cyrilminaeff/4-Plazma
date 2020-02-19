# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:54:23 2019

@author: cyril
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import math
from math import exp, pi
from scipy import integrate
import time

start_time = time.time()

#-----------------------------------------------------------------------------
# Исходные данные задачи

N = 1  # количество частиц каждого сорта

V0_e , V0_i = 100 , 50       # среднеквадратическое отклонение от среднего
k = 3                       # if k=1 P=0.682 / k=2 P=0.954 / k=3 P=0.9973 где P-вероятность того, что V лежит в диапазоне (-Vmax,Vmax)

eps = 0.5           # амплитудный параметр для функции распределения n_alpha
l = 1               # частотный параметр для функции распределения n_alpha
m_i = 1             # относительная масса ионов

n_max = 30          # количество шагов по времени
dt = 1              # шаг по времени 
L = 100             # длина области с плазмой (для N=1 принять L=100 ; N=10000 L=100000)
LL = 100            # длина расчетной области (для N=1 принять L=100 ; N=10000 L=100000)
M = 256             # количество узлов на расчетной области LL
N_int = 1000 + 1    # количество разбиений для расчета интегралов f_alpha и n_alpha

Xj = np.linspace(0 , LL , M)        # равномерная сетка для расчетной области LL
dx = Xj[1]-Xj[0]

kn = np.zeros(int(M/2)+1)           # массив волновых чисел    
for n in range(0,int(M/2)+1):
    kn[n] = n* 2*pi / LL

if N == 1:          # отслеживание положений и скоростей двух частиц
    xe, xi = np.zeros(n_max+1) , np.zeros(n_max+1)
    ve, vi = np.zeros(n_max+1) , np.zeros(n_max+1)

if N >= 1000:        # подсчет кинетической энергии системы
    E_kin = np.zeros(n_max+1)

#-----------------------------------------------------------------------------

# ненормированная функция распределения скорости
def f_nenorm(V, V0_alpha):
    return exp(- V**2 / 2 / V0_alpha**2)

# нормированная функция распределения скорости
def f_alpha(V, V0_alpha, f0_alpha):
    return f0_alpha * f_nenorm(V, V0_alpha)

# вычисление нормировочного множителя f0_e и f0_p
# PS: если бесконечные пределы, то np.inf
f0_e = 1 / integrate.quad(f_nenorm, -V0_e*k, V0_e*k, (V0_e))[0]     
f0_i = 1 / integrate.quad(f_nenorm, -V0_i*k, V0_i*k, (V0_i))[0] 

#-----------------------------------------------------------------------------

# ненормированная функция распределения координаты
def n_nenorm(x, eps, l, L):
    return ( 1 + eps*math.cos(2*pi*l*x/L) )

# нормированная функция распределения скорости
def n_alpha(x, eps, l, L, n0_alpha):
    return n0_alpha * n_nenorm(x, eps, l, L)

# вычисление нормировочного множителя n0_e и n0_p
n0_e = 1 / integrate.quad(n_nenorm, 0, L, (eps, l, L))[0]
n0_i = n0_e

#-----------------------------------------------------------------------------
# создание рандомного массива фазовых координат: 0-ой столбец - х / 1-ый столбец - V
def create_ar_U(N):
    U = np.zeros((N,2))
    for i in range(N):
        U[i,0] = random.random()
        U[i,1] = random.random()
    return U

U_e = create_ar_U(N)
U_i = create_ar_U(N)
 
#-----------------------------------------------------------------------------
# задание каждой из N частиц скорости в соответствии с функцией распределения f_alpha 
def setting_V_for_U (U, f0_alpha, V0_alpha, k, N_int): 
# U - массив, f0_alpha - норм. коэф-т, V0_alpha - дисперсия скорости
    # сортировка массива по скоростям
    Vorder = U[:, 1].argsort()      # массив номеров элементов, сортированных по порядку для столбца U[:, 1]
    U = np.take(U, Vorder, 0)       # отсортированный массив в соответствии с Vorder

    # интегрирование функции расределения f_alpha
    Vmax = V0_alpha * k                         # пределы интегрирования f_alpha
    Vi = np.linspace(-Vmax , Vmax , N_int)      # равномерная сетка скорости
    dVi = Vi[1]-Vi[0]                           # шаг по сетке скорости
    Sum_int = 0                                 # переменная для суммы интеграла f_alpha
    n_start = 0                                 # номер элемента с которого проверяем, что U[j, 1] > Sum_int
    for i in range(len(Vi)-1):
        Sum_int = Sum_int + ( f_alpha(Vi[i], V0_alpha, f0_alpha) + f_alpha(Vi[i+1], V0_alpha, f0_alpha) )/2*dVi
    
        for j in range(n_start , N):
            if U[j, 1] < Sum_int:
                U[j, 1] = (Vi[i]+Vi[i+1])/2
            else:
                n_start = j
                break

    #print(Sum_int, ' (интеграл f_alpha методом трапций)')
    return U


U_e = setting_V_for_U (U_e, f0_e, V0_e, k, N_int)
#print(integrate.quad(f_alpha, -V0_e*k, V0_e*k, (V0_e, f0_e))[0], ' (интеграл f_e по scipy)' )

U_i = setting_V_for_U (U_i, f0_i, V0_i, k, N_int)
#print(integrate.quad(f_alpha, -V0_i*k, V0_i*k, (V0_i, f0_i))[0], ' (интеграл f_i по scipy)' )


#-----------------------------------------------------------------------------
# задание каждой из N частиц координаты в сооветствии с функцией распределения n_alpha
def setting_X_for_U (U, N_int, eps, l, L, n0_alpha):
# U - массив, N_int - количество разбиений интеграла, n0_alpha - норм. коэф-т
    
    # сортировка массива по скоростям
    Xorder = U[:, 0].argsort()      # массив номеров элементов, сортированных по порядку для столбца U[:, 0]
    U = np.take(U, Xorder, 0)       # отсортированный массив в соответствии с Xorder

    # интегрирование функции распределения n_alpha
    Xi = np.linspace(0 , L , N_int)             # равномерная сетка координаты
    dXi = Xi[1]-Xi[0]                           # шаг по сетке координаты
    Sum_int = 0                                 # переменная для суммы интеграла n_alpha
    n_start = 0                                 # номер элемента с которого проверяем, что U[j, 0] > Sum_int
    for i in range(len(Xi)-1):
        Sum_int = Sum_int + ( n_alpha(Xi[i], eps, l, L, n0_alpha) + n_alpha(Xi[i+1], eps, l, L, n0_alpha) )/2*dXi
    
        for j in range(n_start , N):
            if U[j, 0] < Sum_int:
                U[j, 0] = (Xi[i]+Xi[i+1])/2
            else:
                n_start = j
                break

    #print(Sum_int, ' (интеграл n_alpha методом трапций)')
    return U


U_e = setting_X_for_U (U_e, N_int, eps, l, L, n0_e)
#print(integrate.quad(n_alpha, 0, L, (eps, l, L, n0_e))[0], ' (интеграл n_e по scipy)' )    

U_i = setting_X_for_U (U_i, N_int, eps, l, L, n0_i)
#print(integrate.quad(n_alpha, 0, L, (eps, l, L, n0_i))[0], ' (интеграл n_i по scipy)' )  

#-----------------------------------------------------------------------------

def V_plot(U, V0_alpha, k, f0_alpha, N_int):
    
    Vmax = V0_alpha * k                         # пределы интегрирования f_alpha
    Vi = np.linspace(-Vmax , Vmax , N_int)      # равномерная сетка скорости

    # Гистограмма для скоростей
    N_hist = 50                        # количество шагов для гистограммы (не ставить больше, чем N_int, иначе на гистограмме будут пробелы)
    plt.hist(U[:,1], N_hist, label = 'гистограмма V' )
    
    # График функции распределения скорости f_alpha(V)
    plt.plot(Vi, [ N*(2*Vmax/N_hist) * f_alpha(i,V0_alpha,f0_alpha) for i in Vi], label = 'N * dV * f_alpha(V,0)');
    plt.title('Распределение скоростей')
    plt.legend();
    plt.grid()
    plt.xlabel('V')
    plt.ylabel('f(V)')
    plt.xlim(-1.5*Vmax, 1.5*Vmax)
    plt.ylim(bottom=0)
    #plt.savefig(name + '.png', fmt='png', dpi=200)
    plt.show();


V_plot(U_e, V0_e, k, f0_e, N_int)
V_plot(U_i, V0_i, k, f0_i, N_int)

#-----------------------------------------------------------------------------
# интерполяция зарядов в узлы сетки
def interpol_U_in_nj (U, Xj):
    nj = np.zeros((M))
    dx = Xj[1]-Xj[0]
    
    for i in range(N):                  # перебор по всем частицам
        j = int(U[i,0] // dx)           # номер узла расчетной области
        nj[j] = nj[j] + (Xj[j+1]-U[i,0])/dx**2
        nj[j+1] = nj[j+1] + (U[i,0]-Xj[j])/dx**2
    
    # проверка числа частиц после интерполирования
    #inter_N = 0
    #for j in range(M):
    #    inter_N = inter_N + nj[j]*dx
    #print(inter_N, ' (проверка числа частиц)')
    
    return nj


# !!! TEST PROBLEM FOR N==1
if N == 1:
    U_e[0,0]=M//3*dx
    U_i[0,0]=2*M//3*dx
    for i in range(N):
        U_e[i,1]=0
        U_i[i,1]=0
# !!! end TEST PROBLEM FOR N==1


nj_e = interpol_U_in_nj(U_e, Xj)
nj_i = interpol_U_in_nj(U_i, Xj)

#-----------------------------------------------------------------------------
def N_plot(U, nj, n0_alpha):

    # Гистограмма для распределения частиц
    N_hist = 50           # количество шагов для гистограммы (не ставить больше, чем N_int, иначе на гистограмме будут пробелы)
    plt.hist(U[:,0], N_hist, range=(0,L), label = 'гистограмма N(x)' )
    
    Xi = np.linspace(0 , L , N_int) 
    
    # График функции распределения частиц
    plt.plot(Xi, [ N*(L/N_hist)*n_alpha(i, eps, l, L, n0_alpha) for i in Xi], label = 'N * dx * n_alpha(x,0)');
    plt.title('Распределение частиц')
    plt.legend();
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('n')
    plt.xlim(0-L/10, L+L/10)
    plt.ylim(bottom=0)
    #plt.savefig(name + '.png', fmt='png', dpi=200)
    plt.show();
'''    
    # График распределения зарядов частиц
    plt.plot(Xj, [ (L/N_hist)*nj[j] for j in range(M)], label = 'nj_alpha(x)*dx', marker = 'o' , markersize = 2);
    plt.title('Распределение частиц')
    plt.legend();
    plt.grid()
    plt.xlabel('Xj')
    plt.ylabel('n_alpha(x)')
    plt.xlim(0-L/10, L+L/10)
    plt.ylim(bottom=0)
    #plt.savefig(name + '.png', fmt='png', dpi=200)
    plt.show();
'''
#N_plot(U_e, nj_e, n0_e)
#N_plot(U_i, nj_i, n0_i)
'''
#Распределение V(x)
plt.hexbin( [U_e[i,0] for i in range (N)] , [U_e[i,1] for i in range(N)] , cmap='jet')
plt.title('Распределение V(x) для частиц сорта е')
plt.xlabel('x')
plt.ylabel('V(x)')
plt.ylim(-V0_e*k,V0_e*k)
plt.show()
plt.hexbin( [U_i[i,0] for i in range (N)] , [U_i[i,1] for i in range(N)] , cmap='jet')
plt.title('Распределение V(x) для частиц сорта ion')
plt.xlabel('x')
plt.ylabel('V(x)')
plt.ylim(-V0_e*k,V0_e*k)
plt.show()
'''

#-----------------------------------------------------------------------------
# Функция результирующей плотности зарядов
rhoj = [nj_i[j] - nj_e[j] for j in range(M)]

# График плотности зарядов частиц
plt.plot(Xj, [ rhoj[j] for j in range(M)], label = 'rhoj(x)', marker = 'o' , markersize = 4);
plt.title('Распределение плотности заряда')
plt.legend();
plt.grid()
plt.xlabel('Xj')
plt.ylabel('rho(x)')
plt.xlim(0-L/10, L+L/10)
#plt.ylim(bottom=0)
#plt.savefig(name + '.png', fmt='png', dpi=200)
plt.show();

#-----------------------------------------------------------------------------
# Фурье-образ результирующей плотности rho(k)

'''
#Медленное преобразование Фурье
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:int(N/2)] * X_odd,
                               X_even + factor[int(N/2):] * X_odd])
'''

rhok = np.fft.rfft(rhoj)

'''
# сдвиг нулевой частоты в центр (если не rfft)
rhok2 = np.zeros(M, dtype = np.complex128)
for i in range(int(M/2)):
    rhok2[i] = rhok[int(M/2)+i]
for i in range(int(M/2),M):
    rhok2[i] = rhok[i-int(M/2)]

# сдвиг нулевой частоты в центр (если не rfft) через numpy
rhok = np.fft.fftshift(rhok)
'''

'''
# Фурье-образ результирующей плотности rhok(k)
plt.plot( kn, [ np.real(rhok[n]) for n in range(len(rhok))], label = 'rhok(kn)', marker = 'o' , markersize = 4);
plt.title('Фурье образ результирующей плотности')
plt.legend();
plt.grid()
plt.xlabel('k')
plt.ylabel('rho(k)')
#plt.xlim()
#plt.ylim(bottom=0)
#plt.savefig(name + '.png', fmt='png', dpi=200)
plt.show();
'''
#-----------------------------------------------------------------------------
# вычисление потенциала Phik(k) и Phij(x)

#Phik = [ rhok[i]/( kn[i] * math.sin(kn[i]*dx/2)/(kn[i]*dx/2) )**2 for i in range(len(kn)) ]
Phik = [ rhok[i]/( 2 * math.sin(kn[i]*dx/2) / dx )**2 for i in range(len(kn)) ]
Phik[0] = 0*Phik[1]
'''
Phij = np.fft.irfft(Phik)

# Фурье-образ потенциала Phik(k)
plt.plot( kn, [ np.abs(Phik[n]) for n in range(len(Phik))], label = 'Phik(kn)', marker = 'o' , markersize = 4);
plt.title('Фурье образ потенциала')
plt.legend();
plt.grid()
plt.xlabel('k')
plt.ylabel('Phi(k)')
#plt.xlim()
#plt.ylim(bottom=0)
#plt.savefig(name + '.png', fmt='png', dpi=200)
plt.show();

# Потенциал электрического поля Phij
plt.plot( Xj , [ np.real(Phij[j]) for j in range(len(Phij))], label = 'Phij(x)', marker = 'o' , markersize = 4);
#plt.plot(Xj , 1/Xj , label = 'Phij(x)' )
#plt.plot(Xj , 1/(Xj-LL) , label = 'Phij(x)')
#plt.plot(Xj , 1/Xj + 1/(Xj-LL) , label = 'Phij(x)')
plt.title('Распределение потенциала электрического поля')
plt.legend();
plt.grid()
plt.xlabel('Xj')
plt.ylabel('Phi(x)')
plt.xlim(0-LL/10, LL+LL/10)
#plt.ylim(-0.1,0.1)
#plt.savefig(name + '.png', fmt='png', dpi=200)
plt.show();
'''
#-----------------------------------------------------------------------------
# вычисление напряженности Ek(k) и Ej(x)

#Ek = [ -rhok[i]/( kn[i] * math.sin(kn[i]*dx/2)/(kn[i]*dx/2) )**2 * (kn[i]*math.sin(kn[i]*dx)/(kn[i]*dx)) for i in range(len(kn)) ]
Ek = [ -1j * Phik[i] * ( math.sin(kn[i]*dx) / dx ) for i in range(len(kn)) ]
Ek[0] = 0*Ek[1]

Ej = np.fft.irfft(Ek)

'''
# Фурье-образ напряженности электрического поля Ek
plt.plot( kn, [ np.real(Ek[n]) for n in range(len(Ek))], label = 'Ek(kn)', marker = 'o' , markersize = 4);
plt.title('Фурье образ напряженности электрического поля')
plt.legend();
plt.grid()
plt.xlabel('k')
plt.ylabel('E(k)')
#plt.xlim()
#plt.ylim(bottom=0)
#plt.savefig(name + '.png', fmt='png', dpi=200)
plt.show();


# Распределение напряженности электрического поля Ej
plt.plot( Xj , [ np.real(Ej[j]) for j in range(len(Ej))], label = 'Ej(x)', marker = 'o' , markersize = 4);
#plt.plot(Xj , 1/Xj**2 , label = 'Ej(x)' )
#plt.plot(Xj , -1/(Xj-LL)**2 , label = 'Ej(x)'  )
#plt.plot(Xj , -1/Xj**2 + 1/(Xj-LL)**2 , label = 'Ej(x)' )
plt.title('Распределение напряженности электрического поля')
plt.legend();
plt.grid()
plt.xlabel('Xj')
plt.ylabel('E(x)')
plt.xlim(0-LL/10, LL+LL/10)
#plt.ylim(-0.05,0.05)
#plt.savefig(name + '.png', fmt='png', dpi=200)
plt.show();
'''
#-----------------------------------------------------------------------------
# Проверка правильности Ej
'''
Ej2 = np.zeros(M)

# периодические ГУ
Ej2[0] = -(Phij[1]-Phij[M-1])/2/dx
Ej2[M-1] = -(Phij[0]-Phij[M-2])/2/dx
for i in range(1,M-1):
    Ej2[i] = -(Phij[i+1]-Phij[i-1])/2/dx


# Распределение напряженности электрического поля Ej
plt.plot( Xj , [ Ej2[j] for j in range(len(Ej))], label = 'Ej(x)', marker = 'o' , markersize = 4);
#plt.plot(Xj , 1/Xj**2 , label = 'Ej(x)' )
#plt.plot(Xj , -1/(Xj-LL)**2 , label = 'Ej(x)'  )
#plt.plot(Xj , -1/Xj**2 + 1/(Xj-LL)**2 , label = 'Ej(x)' )
plt.title('Распределение напряженности электрического поля')
plt.legend();
plt.grid()
plt.xlabel('Xj')
plt.ylabel('E(x)')
plt.xlim(0-LL/10, LL+LL/10)
#plt.ylim(-0.05,0.05)
#plt.savefig(name + '.png', fmt='png', dpi=200)
plt.show();
'''
#-----------------------------------------------------------------------------
# Интерполяция напряженности Ej из узлов сетки Xj на частицы U[i,0]

def interpol_Ej_to_Ei (U, Ej, Xj):
    E_alpha = np.zeros(N)
    dx = Xj[1]-Xj[0]

    for i in range(N):                  # перебор по всем частицам
        j = int(U[i,0] // dx)           # номер узла расчетной области
        E_alpha[i] = Ej[j]*(Xj[j+1]-U[i,0])/dx + Ej[j+1]*(U[i,0]-Xj[j])/dx
    return E_alpha
    
E_e = interpol_Ej_to_Ei (U_e, Ej, Xj)
E_i = interpol_Ej_to_Ei (U_i, Ej, Xj)

#-----------------------------------------------------------------------------
# Счет на временном слое №1
def  leap_frog_first_step(U, E, N, m, q):    
    #x_14 = [ U[i,0] + (dt/4)*U[i,1] for i in range(N)]
    u_12 = [ U[i,1] + (dt/2)*(q/m)*E[i] for i in range(N)]
    u_32 = [ u_12[i] + (q/m) * E[i] * dt for i in range(N)]
    for i in range(N):
        U[i,0] = U[i,0] + u_32[i]*dt
        U[i,1] = 0.5*( u_12[i] + u_32[i] )
    return U, u_32

def  leap_frog(U, E, N, m, q, u_12):    
    u_32 = [ u_12[i] + (q/m) * E[i] * dt for i in range(N)]
    for i in range(N):
        U[i,0] = U[i,0] + u_32[i]*dt
        U[i,1] = 0.5*( u_12[i] + u_32[i] )
    return U, u_32


def periodic_transfer(U, LL, N):
    for i in range(N):
        if U[i,0] > LL :
            U[i,0] = U[i,0] - LL*(U[i,0] // LL)
        if U[i,0] < 0 :
            U[i,0] = U[i,0] + LL*(1 + (-U[i,0]) // LL)
    return U

# Датчик ШАГ 0
if N == 1:          # отслеживание положений и скоростей двух частиц
    xe[0], xi[0], ve[0], vi[0] = U_e[0,0], U_i[0,0], U_e[0,1], U_i[0,1] 

if N >= 1000:        # подсчет кинетической энергии системы
    for i in range(N):
        E_kin[0] = E_kin[0] + 1/2 * U_e[i,1]**2 + 1/2 * m_i * U_i[i,1]**2


U_e, u_e_32 = leap_frog_first_step(U_e, E_e, N, 1, -1)
U_i, u_i_32 = leap_frog_first_step(U_i, E_i, N, m_i, 1)

U_e = periodic_transfer(U_e, LL, N)
U_i = periodic_transfer(U_i, LL, N)

# Датчик ШАГ 1
if N == 1:          # отслеживание положений и скоростей двух частиц
    xe[1], xi[1], ve[1], vi[1] = U_e[0,0], U_i[0,0], U_e[0,1], U_i[0,1] 

if N >= 1000:        # подсчет кинетической энергии системы
    for i in range(N):
        E_kin[1] = E_kin[1] + 1/2 * U_e[i,1]**2 + 1/2 * m_i * U_i[i,1]**2

'''
#Распределение V(x) на шаге №1
plt.hexbin( [U_e[i,0] for i in range (N)] , [U_e[i,1] for i in range(N)] , cmap='jet')
plt.title('Распределение V(x) для частиц сорта е')
plt.xlabel('x')
plt.ylabel('V(x)')
plt.ylim(-V0_e*k,V0_e*k)
plt.show()
plt.hexbin( [U_i[i,0] for i in range (N)] , [U_i[i,1] for i in range(N)] , cmap='jet')
plt.title('Распределение V(x) для частиц сорта ion')
plt.xlabel('x')
plt.ylabel('V(x)')
plt.ylim(-V0_e*k,V0_e*k)
plt.show()
'''
#-----------------------------------------------------------------------------
# Счет на каждом слое по времени
for n in range(2 , n_max+1):
    nj_e = interpol_U_in_nj(U_e, Xj)
    nj_i = interpol_U_in_nj(U_i, Xj)
    
    rhoj = [nj_i[j] - nj_e[j] for j in range(M)]
    rhok = np.fft.rfft(rhoj)
    
    Phik = [ rhok[i]/( 2 * math.sin(kn[i]*dx/2) / dx )**2 for i in range(len(kn)) ]
    Phik[0] = 0*Phik[1]
    
    Ek = [ -1j * Phik[i] * ( math.sin(kn[i]*dx) / dx ) for i in range(len(kn)) ]
    Ek[0] = 0*Ek[1]
    Ej = np.fft.irfft(Ek)
    
    E_e = interpol_Ej_to_Ei (U_e, Ej, Xj)
    E_i = interpol_Ej_to_Ei (U_i, Ej, Xj)
    
    U_e, u_e_32 = leap_frog(U_e, E_e, N, 1, -1, u_e_32)
    U_i, u_i_32 = leap_frog(U_i, E_i, N, m_i, 1, u_i_32)
    
    U_e = periodic_transfer(U_e, LL, N)
    U_i = periodic_transfer(U_i, LL, N)

    # Датчик ШАГ n
    if N == 1:          # отслеживание положения и скорости для двух частиц
        xe[n], xi[n], ve[n], vi[n] = U_e[0,0], U_i[0,0], U_e[0,1], U_i[0,1] 

    if N >= 1000:        # подсчет кинетической энергии системы
        for i in range(N):
            E_kin[n] = E_kin[n] + 1/2 * U_e[i,1]**2 + 1/2 * m_i * U_i[i,1]**2
    
    nj_e = interpol_U_in_nj(U_e, Xj)
    nj_i = interpol_U_in_nj(U_i, Xj)
    rhoj = [nj_i[j] - nj_e[j] for j in range(M)]
    
    if n_max//4>0 and n%(n_max//4) == 0:   # сюда можно вставить 3 графика
        pass


if N == 1:
    # Изменение положений двух частиц во времени
    plt.plot( [n for n in range(n_max+1)] , [ xe[n] for n in range(n_max+1)], label = 'x_e(t)', marker = 'o' , markersize = 0);
    plt.plot( [n for n in range(n_max+1)] , [ xi[n] for n in range(n_max+1)], label = 'x_i(t)', marker = 'o' , markersize = 0);
    plt.title('Изменение положений двух частиц во времени')
    plt.legend();
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('x(t)')
    #plt.xlim(0-LL/10, LL+LL/10)
    #plt.ylim(-0.05,0.05)
    #plt.savefig(name + '.png', fmt='png', dpi=200)
    plt.show();
    
    # Изменение скоростей двух частиц во времени
    plt.plot( [n for n in range(n_max+1)] , [ ve[n] for n in range(n_max+1)], label = 'v_e(t)', marker = 'o' , markersize = 0);
    plt.plot( [n for n in range(n_max+1)] , [ vi[n] for n in range(n_max+1)], label = 'v_i(t)', marker = 'o' , markersize = 0);
    plt.title('Изменение скоростей двух частиц во времени')
    plt.legend();
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('v(t)')
    #plt.xlim(0-LL/10, LL+LL/10)
    #plt.ylim(-0.05,0.05)
    #plt.savefig(name + '.png', fmt='png', dpi=200)
    plt.show();


if N >= 1000:
    
    V_plot(U_e, V0_e, k, f0_e, N_int)
    V_plot(U_i, V0_i, k, f0_i, N_int)
    N_plot(U_e, nj_e, n0_e)
    N_plot(U_i, nj_i, n0_i)
    

    # Изменение энергии системы во времени
    plt.plot( [n for n in range(n_max+1)] , [ E_kin[n] for n in range(n_max+1)], label = 'E_kin(t)', marker = 'o' , markersize = 0);
    plt.title('Изменение энергии системы во времени')
    plt.legend();
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('E_kin(t)')
    #plt.xlim(0-LL/10, LL+LL/10)
    #plt.ylim(-0.05,0.05)
    #plt.savefig(name + '.png', fmt='png', dpi=200)
    plt.show();

    
    #Распределение V(x)
    plt.hexbin( [U_e[i,0] for i in range (N)] , [U_e[i,1] for i in range(N)] , cmap='jet')
    plt.title('Распределение V(x) для частиц сорта е')
    plt.xlabel('x')
    plt.ylabel('V(x)')
    plt.ylim(-V0_e*k,V0_e*k)
    plt.show()
    plt.hexbin( [U_i[i,0] for i in range (N)] , [U_i[i,1] for i in range(N)] , cmap='jet')
    plt.title('Распределение V(x) для частиц сорта ion')
    plt.xlabel('x')
    plt.ylabel('V(x)')
    plt.ylim(-V0_e*k,V0_e*k)
    plt.show()



#-----------------------------------------------------------------------------
'''
random.random()
random.randint(0,360)
#plt.plot(x[n], F[n][:Nx+1], label = 't = ' + str(round(Timelimit/4*j , 6)) , linewidth=0.5 , marker = 'o' , markersize = 1 );
'''

print("----%s seconds----" % (time.time()-start_time))