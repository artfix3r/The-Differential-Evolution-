import subprocess, re, scipy, json
from pprint import pprint

# ----------------------------------------------------------------
def htest(testf, data0,data1, p=0.05):
	stat,p = testf(data0,data1)
	msg    = 'probably the same distribution' if 0.05<p else 'probably different distributions'
	print(f"\x1b[32m{str(testf).split()[1]:8}  \x1b[0m{msg:32}  stat {stat:6.3f}  p {p:.3f}")

def test(program='cudaDE2022/programDE'):
	print(f'\n----------------------------------------------------------------\n\x1b[35m{program}\x1b[0m')
	reply = subprocess.run(program, stdout=subprocess.PIPE, shell=True)
	data  = json.loads(reply.stdout)
	for i in range(3):
		data0 = data[3*i+0]['x_mean']
		data1 = data[3*i+1]['x_mean']
		print()
		print(f"dim \x1b[31m{data[3*i]['dim']}  \x1b[0mpop \x1b[32m{data[3*i]['pop']}  \x1b[0mgen \x1b[34m{data[3*i]['gen']}\x1b[0m")
		print('data0', data0)
		print('data1', data1)
		htest(scipy.stats.wilcoxon, data0,data1)
		htest(scipy.stats.kruskal,  data0,data1)

# ----------------------------------------------------------------
test('cudaDE2022/programDE')
test('cudaPSO2022/pso')

# ----------------------------------------------------------------
# import numpy np
# data0 = np.random.normal(0,1, 0x10)  # np.random.normal(0,1, 0x10)
# data1 = np.random.normal(0,2, 0x10)  # np.random.normal(0,2, 0x10)
# htest(scipy.stats.wilcoxon, data0,data1)  # wilcoxon signed-rank test
# htest(scipy.stats.kruskal,  data0,data1)  # kruskal-wallis h     test
