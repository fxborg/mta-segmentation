from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D

import numpy as np
from pprint import pprint
import math
import sys

rel_rms_improvement = 0.05
max_depth_of_tree = 10
max_num_of_intervals = 5

def residuals(arr_x,arr_y):
	A = np.ones((len(arr_x),2),float)
	A[:,0]=arr_x
	(p,residuals,rank,s) = np.linalg.lstsq(A,arr_y)
	try:
		err = residuals[0]
	except IndexError:
		err = 0.0
	
	return err

def get_rms(arr_x, arr_y, epoches):
	num_epoches = len(epoches)
	error_fit = 0.0
	for i in range(num_epoches-1):
		temp_x = arr_x[epoches[i]:epoches[i+1]+1]
		temp_y = arr_y[epoches[i]:epoches[i+1]+1]
		err = residuals(temp_x,temp_y)
		error_fit = error_fit + err 
	return math.sqrt(error_fit)



def get_norm_error(arr_x,arr_y, epoches):
	num_epoches = len(epoches)
	error_zero = math.sqrt(residuals(arr_x,arr_y))
	error_fit = 0.0
	for i in range(num_epoches-1):
		j0=epoches[i]
		j1=epoches[i+1]
		temp_x=arr_x[j0:j1+1]
		temp_y=arr_y[j0:j1+1]
		err = residuals(temp_x,temp_y)
		error_fit=error_fit + err
	
	error_fit = math.sqrt(error_fit)
	if(num_epoches-2>0):
		ret = -math.log(error_fit/error_zero)/ (num_epoches-2)
	else:
		ret = 0.0
	return ret
	
def findmaxmin(arr_x,arr_y,offset):
	coef=np.polyfit(arr_x,arr_y,1)
	sz = len(arr_x)
	
	line = np.polyval(coef,arr_x)
	diff = arr_y -line 
	imax = np.argmax(diff)
	imin = np.argmin(diff)
	epoch=[]
	
	if(imin>1 and imin < sz-2): epoch.append(imin)
	if(imax>1 and imax < sz-2): epoch.append(imax)
	# special case
	if(len(epoch)>0): return (np.array(epoch)+offset).tolist()
	a = (arr_y[sz-1]-arr_y[0])/(arr_x[sz-1]-arr_x[0])
	b = arr_y[0]-a*arr_x[0]
	coef=[a,b]

	line = np.polyval(coef,arr_x)
	diff = arr_y -line 
	imax = np.argmax(diff)
	imin = np.argmin(diff)
	epoch=[]

	if(imin>1 and imin < sz-2): epoch.append(imin)
	if(imax>1 and imax < sz-2): epoch.append(imax)
	return (np.array(epoch)+offset).tolist()

def findepoches(arr_x, arr_y,max_intervals):
	sz = len(arr_x)	
	epoches = [0,sz-1]
	cnt = 0
	while(len(epoches) < max_intervals+1):
		curr_epoch_num = len(epoches)
		new_epoches = []
		for i in range(curr_epoch_num-1):
			j0=epoches[i]
			j1=epoches[i+1]
			new_epoches = findmaxmin(arr_x[j0:j1+1],arr_y[j0:j1+1],j0)
			
			epoches.extend(new_epoches)

		epoches.sort()
		cnt+=1
		if(cnt>10): break
	return epoches

def findapproximation(arr_x, arr_y, max_intervals):
	from itertools import combinations

	epoches = findepoches(arr_x,arr_y, max_intervals)	
	
	sz=len(epoches)
	calc_sz= (sz-2)**2 -1
	combin_list = []
	combin_list.append(epoches)
	
	for i in range(1,sz-1):
		combin_list.append([epoches[0],epoches[i],epoches[sz-1]])

	
	for i in range(2,sz-2):
		combis = list(combinations(epoches[1:sz-1],i))
		sizer = len(combis)
		for j in range(sizer):
			combin_list.append([epoches[0]]+list(combis[j])+[epoches[sz-1]])
			
	
	numepoch = len(combin_list)
	
	error_list = [] 
	for i in range(numepoch):
		error_list.append(get_norm_error(arr_x,arr_y,combin_list[i]))
	
	
	imax = error_list.index(max(error_list))
	final_epoches = combin_list[imax]
	rms_error = get_rms(arr_x,arr_y,final_epoches)
	return [final_epoches,rms_error]


def draw_plot(arr_x,arr_y,plot_title):
	plot(arr_x,arr_y,'-')
	title(plot_title)
	xlabel("X")
	ylabel("Y")
	xlim((0,len(arr_x)))


def draw_line(arr_x,arr_y,epoches):
	sz= len(epoches)
	for i in range(sz-1):
		j0=epoches[i]
		j1=epoches[i+1]
		temp_x= arr_x[j0:j1+1]
		temp_y= arr_y[j0:j1+1]
		a,b = np.polyfit(temp_x,temp_y,1)
		line_x=[arr_x[j0],arr_x[j1]]
		line_y=[a*arr_x[j0]+b,a*arr_x[j1]+b]
		plot(line_x,line_y,'ro-')

if __name__ == "__main__":
	arr_xy=np.loadtxt("./testdata/sample.dat", skiprows=1)
	arr_x=np.asarray(arr_xy[:, 0])
	arr_y=np.asarray(arr_xy[:, 1])
	#arr_x=np.asarray([0,1,2,3,4])
	#arr_y=np.asarray([10,25,40,35,50])
	approx_tree = [[]] * max_depth_of_tree
	rms_tree =	np.zeros(max_depth_of_tree)
	segments_number = np.zeros(max_depth_of_tree)
	arrlength =len(arr_x)
	epoches=[0, arrlength-1]
	approx_tree[0] = epoches
		 
	#// top level 
	rms_tree[0] = get_rms(arr_x,arr_y,epoches)
	for i in range(1,max_depth_of_tree): 
		prev_epoch = approx_tree[i-1] # 直前のセグメント
		curr_epoches_num = len(prev_epoch) # 分割数
		rms_segments = [0.0]*(curr_epoches_num-1)
		new_epoches=[[]] * (curr_epoches_num-1)
		# ---
		
		for j in range(curr_epoches_num-1):
			prev_begin = prev_epoch[j]
			prev_end = prev_epoch[j+1]

			if(prev_end-prev_begin>2):
				[epoches,rms_segments[j]] = findapproximation(
												arr_x[prev_begin:prev_end+1],
												arr_y[prev_begin:prev_end+1],
												max_num_of_intervals)
				
				
				if prev_begin>0:
					epoches = [v + prev_begin for v in epoches]
				new_num = len(epoches)
				epoches = prev_epoch + epoches[1:new_num-1]
				epoches.sort()
				rms_segments[j] = get_rms(arr_x,arr_y,epoches)
				new_epoches[j]=epoches
			else:
		
				new_epoches[j]=prev_epoch
				rms_segments[j]=rms_tree[i-1]
		
		imin = rms_segments.index(min(rms_segments))

		approx_tree[i] = new_epoches[imin]
		segments_number[i] = len(new_epoches[imin])-1
		rms_tree[i]= rms_segments[imin]
		
		
	
	rms_plot = rms_tree * (1.0/rms_tree[0])
		
	opt_level=-1
	i=0;
	sz=len(rms_plot)
	for i in range(sz-1):
		if(rms_plot[i]-rms_plot[i+1] < rel_rms_improvement):
			opt_level=i
			break;
	
	optimal_epoches = approx_tree[opt_level]

	draw_plot(arr_x,arr_y,"MTA analysis")
	draw_line(arr_x,arr_y,optimal_epoches)
	show()
