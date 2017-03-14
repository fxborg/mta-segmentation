from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D

import numpy as np
from pprint import pprint
import math
import sys
from onlineregression import OnlineRegression


rel_rms_improvement = 0.05  # フィットの精度
max_depth_of_tree = 10      # 計算する深さ
max_num_of_intervals = 5    # 二次トレンドの最大数
zigzag_depth = 4            # 最小セグメントの判定に使用する期間

stats = {} #データ
cached_x = []  #
cached_y = []  #
""" 与えられた値より小さい位置を返す"""
def get_prev_index(idx, v):
	idx_sz=len(idx)
	prev=-1
	for j in reversed(range(idx_sz)):
		if v > idx[j]:
			prev=j
			break
	return prev


def find_vertex(arr_x,arr_y,offset,depth):
	sz = len(arr_x)
	a = (arr_y[sz-1]-arr_y[0])/(arr_x[sz-1]-arr_x[0])
	b = arr_y[0]-a*arr_x[0]
	coef=[a,b]
	line = np.polyval(coef,arr_x)
	diff = abs(arr_y -line) 
	imax = np.argmax(diff)
		
	if(imax >=depth  and imax < sz-depth) : return offset+imax
	return None


def add_vertex( arr_x, arr_y, stats , cached_x, cached_y,  seg):
	# 新しい列を作成
	stats[seg]=dict()
	stats[seg][seg] = OnlineRegression()
	stats[seg][seg].push(arr_x[seg],arr_y[seg])
	
	p = get_prev_index(cached_x,seg)
	prev = cached_x[p] if p >= 0 else -1
	
	# 対象期間の値を加算
	temp = OnlineRegression()
	for j in range(prev+1, seg+1): 
		temp.push(arr_x[j], arr_y[j])

	# 直前のデータがあれば行に合計値をセット
	if p >= 0:
		for k,v in stats[prev].items():
			stats[seg][k] = v + temp

	# 以降にデータがあれば
	sz=len(cached_x)
	if p+1 < sz:
		last = cached_x[sz-1]
		temp=OnlineRegression()
		for j in range(seg,last+1):
			temp.push(arr_x[j], arr_y[j])
			if j in stats:
				stats[j][seg]=temp

	# インデックスを更新
	i_ins=p+1
	cached_x.insert(i_ins,seg)
	cached_y.insert(i_ins,arr_y[seg])


def residuals(arr_x,arr_y):
	"""残差平方和を求める"""
	A = np.ones((len(arr_x),2),float)
	A[:,0]=arr_x
	(p,residuals,rank,s) = np.linalg.lstsq(A,arr_y)
	try:
		err = residuals[0]
	except IndexError:
		err = 0.0
	
	return err

def get_rms(stats, epoches):
	"""区間分割後の誤差を求める"""
	num_epoches = len(epoches)
	error_fit = 0.0
	for i in range(num_epoches-1):
		err = stats[epoches[i+1]][epoches[i]].residuals()
		error_fit = error_fit + err 

	return math.sqrt(error_fit)



def get_norm_error(arr_x,arr_y, epoches):
	"""
	ペーパー '2.2 Optimal piecewise linear approximation'の評価式
	
	"""

	num_epoches = len(epoches)
	error_zero = math.sqrt(stats[epoches[-1]][epoches[0]].residuals())

	error_fit = 0.0
	for i in range(num_epoches-1):
		err = stats[epoches[i+1]][epoches[i]].residuals()
		error_fit = error_fit + err 

	error_fit = math.sqrt(error_fit)
	
	if(num_epoches-2>0):
		ret = -math.log(error_fit/error_zero)/ (num_epoches-2)
	else:
		ret = 0.0
	return ret
	
def findmaxmin(arr_x,arr_y):
	"""残差が最大最小となるポイントを求める"""
	sz=len(arr_x)
	ifm=arr_x[0]
	ito=arr_x[-1]
	a = stats[ito][ifm].slope()
	b = stats[ito][ifm].intercept()
	coef=[a,b]
	line = np.polyval(coef,arr_x)

	diff = arr_y -line 
	imax = np.argmax(diff)
	imin = np.argmin(diff)
	epoch=[]
	if(imin>0 and imin < sz-1): epoch.append(arr_x[imin])
	if(imax>0 and imax < sz-1): epoch.append(arr_x[imax])

	# special case
	if(len(epoch)>0): return epoch
	a = (arr_y[-1]-arr_y[0])/(arr_x[-1]-arr_x[0])
	b = arr_y[0] - a * arr_x[0]

	coef=[a,b]
	line = np.polyval(coef,arr_x)
	diff = arr_y -line 
	imax = np.argmax(diff)
	imin = np.argmin(diff)
	epoch=[]

	if(imin>0 and imin < sz-1): epoch.append(arr_x[imin])
	if(imax>0 and imax < sz-1): epoch.append(arr_x[imax])
	return epoch

def findepoches(arr_x, arr_y,max_intervals):
	"""区間を求める"""
	epoches = [arr_x[0], arr_x[-1]]
	cnt = 0
	while(len(epoches) < max_intervals+1):
		curr_epoch_num = len(epoches)
		new_epoches = []
		for i in range(curr_epoch_num-1):
			j0=epoches[i]
			j1=epoches[i+1]
			selected = np.where((arr_x >= j0) & (arr_x <= j1))
			new_epoches = findmaxmin(arr_x[selected],arr_y[selected])
			
			epoches.extend(new_epoches)

		epoches.sort()
		cnt+=1
		if(cnt>10): break
	return epoches

def findapproximation(arr_x, arr_y, max_intervals):
	"""可能な区間の組み合わせの中から最適な区間を求める"""
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
	rms_error = get_rms(stats,final_epoches)
	return [final_epoches,rms_error]


def draw_plot(arr_x,arr_y,plot_title):
	"""データをプロット"""
	plot(arr_x,arr_y,'-')
	title(plot_title)
	xlabel("X")
	ylabel("Y")
	xlim((arr_x[0],arr_x[-1]))


def draw_line(arr_x,arr_y,epoches):
	"""フィットしたラインをプロット"""
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
	"""メイン"""
	#ファイルから読込み
	arr_xy=np.loadtxt("./testdata/sample.dat", skiprows=1)
	arr_x=np.asarray(arr_xy[:, 0])
	arr_y=np.asarray(arr_xy[:, 1])
	
	period=zigzag_depth*3
	arr_sz =len(arr_x)

	add_vertex(arr_x, arr_y,stats, cached_x, cached_y,0)	#先頭


	for i  in range(arr_sz):
		if i < period-1:continue
		fm=i-(period-1)
		to=i+1
		v = find_vertex(arr_x[fm:to],arr_y[fm:to],fm, zigzag_depth)
		if v and v not in stats and v-1 not in stats and v+1 not in stats :
			add_vertex(arr_x, arr_y,stats, cached_x, cached_y,v)

	add_vertex(arr_x, arr_y,stats, cached_x, cached_y,i)	#末尾


	cached_x = np.asarray(cached_x)
	cached_y = np.asarray(cached_y)

	#計算用のバッファを初期化
	approx_tree = [[]] * max_depth_of_tree
	rms_tree =	np.zeros(max_depth_of_tree)
	segments_number = np.zeros(max_depth_of_tree)
	epoches=[cached_x[0], cached_x[-1]]
	approx_tree[0] = epoches
		 
	#分割無しの誤差をセット
	rms_tree[0] = get_rms(stats,epoches)

	
	for i in range(1,max_depth_of_tree): 

		prev_epoch = approx_tree[i-1] # 一つ前の分割結果
		curr_epoches_num = len(prev_epoch) # 分割数
		
		rms_segments = [0.0]*(curr_epoches_num-1)
		new_epoches=[[]] * (curr_epoches_num-1)
		
		for j in range(curr_epoches_num-1):
			prev_begin = prev_epoch[j]
			prev_end = prev_epoch[j+1]
			#分割可能ならば
			if(prev_end-prev_begin>2):
				selected = np.where((cached_x>=prev_begin) & (cached_x<=prev_end))

				#最適な区間を求める
				[epoches,rms_segments[j]] = findapproximation(
												cached_x[selected],
												cached_y[selected],
												max_num_of_intervals)

				new_num = len(epoches)
				epoches = prev_epoch + epoches[1:new_num-1]
				epoches.sort()
				
				#最新の誤差と分割結果をセット
				rms_segments[j] = get_rms(stats,epoches)
				new_epoches[j]=epoches
			else:
				new_epoches[j]=prev_epoch
				rms_segments[j]=rms_tree[i-1]

		#誤差が最小となる最適な区間を取得
		imin = rms_segments.index(min(rms_segments))
		approx_tree[i] = new_epoches[imin]
		#区間候補を追加
		segments_number[i] = len(new_epoches[imin])-1
		rms_tree[i]= rms_segments[imin]
	
	
	#区間候補リストの精度を計算
	rms_plot = rms_tree * (1.0/rms_tree[0])
		
	opt_level=-1
	i=0;
	sz=len(rms_plot)
	#条件を満たす精度の区間を取得
	for i in range(sz-1):
		if(rms_plot[i]-rms_plot[i+1] < rel_rms_improvement):
			opt_level=i
			break;
	print("------------ segments -----------")
	pprint(approx_tree)
	print("------------- score -------------")
	pprint(rms_tree)
	optimal_epoches = approx_tree[opt_level]
	#プロット
	draw_plot(arr_x,arr_y,"EURUSD(15M) 3/6 - 3/14 MTA Segmentation")
	draw_line(arr_x,arr_y,optimal_epoches)
	show()