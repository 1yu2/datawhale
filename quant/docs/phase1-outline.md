# Phase 1：量化金融入门

## 一、阶段目标

- 理解量化金融的基本思维
- 使用 Python 获取和分析股票数据
- 计算收益率和简单波动指标
- 使用移动平均线构建交易规则
- 对交易策略进行基础回测
- 从收益和风险两个角度解释回测结果

核心学习链路：

```text
提出假设 → 获取数据 → 分析数据 → 制定规则 → 历史回测 → 解释结果
```

## 二、环境准备

- Python 3
- Jupyter Lab
- pandas、numpy
- matplotlib、seaborn
- yfinance、akshare
- 可正常访问行情数据的网络环境

启动课程：

```bash
cd squant-for-beginners
source .venv/bin/activate
python -m jupyter lab
```

## 三、第一章：什么是量化金融

### 核心知识

- 量化金融的基本定义
  - 量化金融是金融学、数学、统计学与计算机科学的交叉学科。它利用算法、数学模型和海量数据来分析金融市场，广泛应用于资产定价、风险管理和程序化交易。该领域旨在通过严格的数据分析消除人为情绪干扰，优化投资回报。
- 主观交易与量化交易的区别
  - **主观交易**依赖交易员个人的经验、逻辑分析和盘感做出决策；**量化交易**则依靠数学模型和计算机代码，通过回测历史数据并执行自动化策略来获取超额收益。两者核心区别在于“决策依据”与“执行方式”的不同
- 量化研究的四步流程
  - **数据处理**、**因子开发**、**组合优化**与**回测执行**
- Python 在量化研究中的作用
- 随机游走与概率优势
  - **随机游走**（Random Walk）是一个数学统计模型，描述的是没有固定方向、每一步都完全独立的随机轨迹。当它与**概率优势**（Probability Edge）结合时，揭示了一个深刻的核心法则：**在不确定性的随机系统中，哪怕只有微小的概率优势，经过无限次重复，最终也会导致结果的必然倾斜（如著名的“赌徒输光定理”）。**

### 实操任务

- **实验一：模拟布朗运动**
  - 单只股票的走势无法预测——每条路径随机且不重复
  - 大量股票的统计规律是确定的——终点价格服从正态分布，波动√t成正比
  - 不预测单一结果，利用统计规律

```python
# ========== 实验一：布朗运动 / 随机游走 ==========
import numpy as np                       # 数值计算（数组、随机数、统计）
import matplotlib.pyplot as plt            # 绘图库（折线图、柱状图等）

plt.rcParams['font.sans-serif'] = ['Heiti SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']  # 跨平台中文字体回退
plt.rcParams['axes.unicode_minus'] = False      # 坐标轴负号正常显示

np.random.seed(2026)        # 固定随机种子

n_stocks = 50               # 模拟 50 只「虚拟股票」
n_days = 250                # 每只走 250 个交易日（约一年）
start_price = 100           # 起始价都是 100 元
daily_volatility = 0.02     # 日波动强度（标准差约2%）

fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1行3列，三个子图

# --- 左图：画出 50 条价格路径 ---
all_paths = []              # 用来存每只股票的路径
for _ in range(n_stocks):   # 循环 50 次，每只股一条路径
    daily_returns = np.random.normal(0, daily_volatility, n_days)  # 每天随机收益
    price_path = start_price * np.cumprod(1 + daily_returns)       # 价格连乘
    all_paths.append(price_path)          # 存入列表
    axes[0].plot(price_path, alpha=0.3, linewidth=0.8)  # 画在左图，半透明

axes[0].axhline(y=start_price, color='red', linestyle='--', alpha=0.5, label='起始价 100元')  # 参考线
axes[0].set_title('50只虚拟股票的随机游走路径', fontsize=13)  # 设置上图标题
axes[0].set_xlabel('交易日')  # 执行本行代码
axes[0].set_ylabel('价格 (元)')  # 设置上图纵轴
axes[0].legend()  # 显示上图图例
axes[0].grid(True, alpha=0.3)  # 上图显示网格

# --- 中图：一年后的终点价格分布（直方图）---
final_prices = [path[-1] for path in all_paths]   # 取每条路径最后一天的价格
axes[1].hist(final_prices, bins=15, color='steelblue', edgecolor='white', alpha=0.8)  # 画直方图
axes[1].axvline(x=start_price, color='red', linestyle='--', label=f'起始价 {start_price}元')  # 中图画垂直参考线
axes[1].axvline(x=np.mean(final_prices), color='orange', linestyle='-', linewidth=2,  # 中图画垂直参考线
                label=f'平均终点价 {np.mean(final_prices):.1f}元')  # 图例文字
axes[1].set_title('1年后的终点价格分布', fontsize=13)  # 设置下图标题
axes[1].set_xlabel('最终价格 (元)')  # 设置下图横轴（日期）
axes[1].set_ylabel('股票数量')  # 设置下图纵轴
axes[1].legend()                                    # 显示下图图例
axes[1].grid(True, alpha=0.3)  # 下图显示网格

# --- 右图：验证「波动 ∝ 时间的平方根」---
time_points = [5, 10, 20, 50, 100, 150, 200, 250]  # 选取若干观察日
std_at_time = []                                    # 存放每个时点的价格标准差
for t in time_points:  # 代码块开始
    prices_at_t = [path[t-1] for path in all_paths]  # 所有股票在第 t 天的价格
    std_at_time.append(np.std(prices_at_t))         # 算标准差

sqrt_time = np.sqrt(time_points)                    # 时间开平方
scale = std_at_time[0] / sqrt_time[0]               # 缩放系数，让理论线对齐第一个点

axes[2].scatter(time_points, std_at_time, s=80, color='steelblue', zorder=5, label='实际波动(标准差)')  # 右图：散点
axes[2].plot(time_points, scale * sqrt_time, 'r--', linewidth=2, label='理论值: σ ∝ √t')  # 右图：理论曲线
axes[2].set_title('波动率 vs 时间 (验证√t法则)', fontsize=13)  # 设置右图标题
axes[2].set_xlabel('交易天数')  # 执行本行代码
axes[2].set_ylabel('价格标准差 (元)')  # 执行本行代码
axes[2].legend()  # 执行本行代码
axes[2].grid(True, alpha=0.3)  # 透明度

plt.tight_layout()                       # 自动调整子图间距，避免标签被裁切
plt.show()                               # 在 Notebook 里显示图片

# ========== 打印文字结论 ==========
print("=" * 60)  # 打印输出
print("实验结论：")  # 打印输出
print(f"  50只股票起始价均为 {start_price}元")  # 打印价格统计
print(f"  1年后最高价: {max(final_prices):.2f}元")  # 打印价格统计
print(f"  1年后最低价: {min(final_prices):.2f}元")  # 打印价格统计
print(f"  1年后平均价: {np.mean(final_prices):.2f}元")  # 格式化打印
print(f"  价格标准差:  {np.std(final_prices):.2f}元")  # 格式化打印
print("-" * 60)  # 打印输出
print("  → 每条路径完全不可预测（左图）")  # 打印分隔线或结论
print("  → 但大量路径的终点价格呈正态分布（中图）")  # 打印分隔线或结论
print("  → 波动幅度与时间平方根成正比（右图）——Regnault 1860年代的发现！")  # 打印分隔线或结论
print("=" * 60)  # 打印输出

```

![](image/1.png)

- **实验二：索普——概率优势**



```python
# ========== 实验二：索普——概率优势 + 凯利仓位 ==========
import numpy as np                       # 数值计算（数组、随机数、统计）
import matplotlib.pyplot as plt            # 绘图库（折线图、柱状图等）

plt.rcParams['font.sans-serif'] = ['Heiti SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']  # 跨平台中文字体回退
plt.rcParams['axes.unicode_minus'] = False      # 坐标轴负号正常显示

np.random.seed(2026)  # 固定随机种子，结果可复现

n_rounds = 1000             # 每位玩家模拟下注 1000 轮
n_simulations = 200         # 每种策略重复 200 次（看分布）
initial_capital = 1000      # 初始本金 1000 元

win_prob_no_edge = 0.50     # 玩家A：无优势，胜率 50%
win_prob_edge = 0.52        # 玩家B/C：有 2% 概率优势
payout_ratio = 1.0          # 赔率 1:1（赢一倍赌注，输一倍赌注）

# 凯利公式: f* = (bp - q) / b  → 最优下注占本金比例
kelly_fraction = (payout_ratio * win_prob_edge - (1 - win_prob_edge)) / payout_ratio  # 凯利公式最优下注比例
aggressive_fraction = 0.25  # 玩家C：有优势但每次下注 25%（太激进）


def simulate_player(win_prob, bet_fraction, n_sims=n_simulations):  # 定义模拟下注函数
    """模拟多人在 n_rounds 轮里的资金曲线。"""  # 字典字段
    all_curves = []              # 存放每次实验的资金曲线
    for _ in range(n_sims):              # 外层：重复很多次实验
        capital = initial_capital        # 本轮起始资金
        curve = [capital]              # 记录每轮后的资金
        for _ in range(n_rounds):      # 内层：一轮轮下注
            bet = capital * bet_fraction   # 本轮下注额 = 本金 × 比例
            if np.random.random() < win_prob:  # 随机数小于胜率 → 赢
                capital += bet * payout_ratio  # 赢：拿回赌注并赚一倍
            else:                              # 否则 → 输
                capital -= bet                 # 输：输掉本轮赌注
            capital = max(capital, 0.01)       # 防止资金变成负数
            curve.append(capital)  # 执行本行代码
        all_curves.append(curve)  # 执行本行代码
    return np.array(all_curves)        # 转成二维数组：行=实验，列=轮次


curves_A = simulate_player(win_prob_no_edge, kelly_fraction)       # 无优势 + 凯利
curves_B = simulate_player(win_prob_edge, kelly_fraction)          # 有优势 + 凯利
curves_C = simulate_player(win_prob_edge, aggressive_fraction)     # 有优势 + 重仓

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))  # 三个玩家各一张子图

configs = [  # 三个子图的配置
    (curves_A, '玩家A: 无优势 (胜率50% + 凯利仓位)', 'red', f'胜率50%, 仓位{kelly_fraction*100:.1f}%'),  # 执行本行代码
    (curves_B, '玩家B: 索普策略 (胜率52% + 凯利仓位)', 'green', f'胜率52%, 仓位{kelly_fraction*100:.1f}%'),  # 执行本行代码
    (curves_C, '玩家C: 有优势但冒进 (胜率52% + 重仓)', 'goldenrod', f'胜率52%, 仓位25%'),  # 执行本行代码
]                                              # 数组拼接结束

for ax, (curves, title, color, label) in zip(axes, configs):  # 为每个玩家画子图
    for i in range(min(30, n_simulations)):   # 只画前30条细线，避免太乱
        ax.plot(curves[i], alpha=0.15, linewidth=0.6, color=color)  # 在子图上画折线
    median_curve = np.median(curves, axis=0)  # 200次实验的中位数轨迹
    ax.plot(median_curve, color=color, linewidth=2.5, label=f'中位数轨迹 ({label})')  # 在子图上画折线
    ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label=f'初始本金 {initial_capital}元')  # 画水平参考线
    ax.set_title(title, fontsize=12, fontweight='bold')  # 设置子图标题
    ax.set_xlabel('下注轮次')  # 设置子图横轴
    ax.set_ylabel('资金 (元)')  # 设置子图纵轴
    ax.legend(fontsize=9, loc='upper left')  # 显示图例
    ax.grid(True, alpha=0.3)  # 显示网格
    ax.set_yscale('log')      # 纵轴用对数刻度，差距大时更好看
    ax.set_ylim(1, None)  # 设置纵轴范围

plt.tight_layout()                       # 自动调整子图间距，避免标签被裁切
plt.show()                               # 在 Notebook 里显示图片

print("=" * 70)  # 打印输出
print(f"凯利公式计算: 最优下注比例 f* = {kelly_fraction*100:.2f}% (胜率52%, 赔率1:1)")  # 打印凯利公式结果
print("=" * 70)  # 打印输出

for name, curves, color in [("玩家A (无优势)", curves_A, "red"),  # 打印各玩家统计
                              ("玩家B (索普策略)", curves_B, "green"),  # 执行本行代码
                              ("玩家C (有优势但冒进)", curves_C, "goldenrod")]:  # 代码块开始
    finals = curves[:, -1]                    # 每个实验最后一轮的资金
    win_rate = np.mean(finals > initial_capital) * 100  # 最终赚钱的比例
    median_final = np.median(finals)  # 赋值：median_final
    print(f"\n  【{name}】")  # 格式化打印
    print(f"    中位数终点资金: {median_final:,.0f}元")  # 格式化打印
    print(f"    盈利概率: {win_rate:.1f}%")  # 格式化打印
    print(f"    最好情况: {np.max(finals):,.0f}元 | 最差情况: {np.min(finals):,.2f}元")  # 格式化打印

print("\n" + "=" * 70)  # 打印输出
print("  → 没有概率优势，凯利公式也救不了你（玩家A）")  # 打印分隔线或结论
print("  → 仅仅2%的胜率优势 + 科学仓位管理 = 长期稳定复利（玩家B）")  # 打印分隔线或结论
print("  → 有优势但仓位过重，反而可能亏光（玩家C）")  # 打印分隔线或结论
print("  → 这就是索普的核心发现：概率优势 × 仓位管理 = 量化盈利的底层公式")  # 打印分隔线或结论
print("=" * 70)  # 打印输出

```

![](image/2.png)

- 使用代码下载 AAPL 行情

```python
# ========== 第一章爽点：下载真实股票数据 ==========
import pandas as pd                  # 处理 Yahoo 返回的行情数据
import matplotlib.pyplot as plt     # 画图
from curl_cffi import requests      # 发起兼容浏览器 TLS 的网络请求

plt.rcParams['font.sans-serif'] = ['Heiti SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']  # 跨平台中文字体回退
plt.rcParams['axes.unicode_minus'] = False      # 坐标轴负号正常显示

# 直接请求 Yahoo Chart 接口，避开被广告规则拦截的 cookie 端点
def download_yahoo_chart(ticker, period='6mo', interval='1d'):
    response = requests.get(
        f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}',
        params={
            'range': period,
            'interval': interval,
            'events': 'div,splits',
        },
        impersonate='chrome',
        timeout=20,
    )
    response.raise_for_status()

    chart = response.json()['chart']
    error = chart['error']
    if error is not None or not chart['result']:
        raise RuntimeError(f'Yahoo 行情下载失败: {error}')

    result = chart['result'][0]
    quote = result['indicators']['quote'][0]
    dates = (
        pd.to_datetime(result['timestamp'], unit='s', utc=True)
        .tz_convert('America/New_York')
        .tz_localize(None)
        .normalize()
    )

    data = pd.DataFrame({
        'Open': quote['open'],
        'High': quote['high'],
        'Low': quote['low'],
        'Close': quote['close'],
        'Volume': quote['volume'],
    }, index=dates)
    data.index.name = 'Date'
    data = data.dropna(subset=['Close'])

    if data.empty:
        raise RuntimeError(f'{ticker} 没有返回有效行情')

    return data


# 下载苹果 AAPL 最近 6 个月的日线（需要联网）
aapl = download_yahoo_chart('AAPL', period='6mo')

print('🎉 恭喜！你已经拿到真实股票数据')  # 打印输出
print(f'   共 {len(aapl)} 个交易日')                              # 行数 = 交易日个数
print(f'   最新收盘价: ${aapl["Close"].iloc[-1]:.2f}')            # iloc[-1] = 最后一行
display(aapl.tail(5))   # 在 Notebook 里美观地显示最后 5 行表格

# ========== 上图收盘价、下图成交量 ==========
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,       # 2行子图，横轴对齐
                         gridspec_kw={'height_ratios': [3, 1]})    # 上图占 3 份高度
axes[0].plot(aapl.index, aapl['Close'], color='tab:blue', linewidth=1.5)  # 折线：收盘价
axes[0].set_title('真实数据 · 苹果 AAPL 收盘价', fontsize=14)  # 设置上图标题
axes[0].set_ylabel('美元')  # 设置上图纵轴
axes[0].grid(True, alpha=0.3)  # 上图显示网格
axes[1].bar(aapl.index, aapl['Volume'], width=0.8, color='gray', alpha=0.5)  # 柱状：成交量
axes[1].set_ylabel('成交量')  # 设置下图纵轴
axes[1].set_xlabel('日期')  # 设置下图横轴（日期）
axes[1].grid(True, alpha=0.3)  # 下图显示网格

plt.tight_layout()                       # 自动调整子图间距，避免标签被裁切
plt.show()                               # 在 Notebook 里显示图片

```

![](image/3.png)

- 观察股票价格和成交量
- 将股票替换为 TSLA 或 NVDA

![](image/4.png)

- 判断近 6 个月的整体走势

### 笔记产出

- 一段自己对“量化金融”的解释
- 一张价格与成交量图
- 最新收盘价和趋势判断
- 对“量化是否等于预测未来”的回答

## 四、第二章：收益率与数据分析

### 核心知识

- OHLCV 数据结构
- 收盘价与复权价格
- 简单收益率
- 收益率曲线和直方图
- 使用标准差比较股票波动

### 关键代码

- pct_change()
- describe()
- std()
- plot()
- hist()

### 实操任务

- 计算股票日收益率
- 绘制收益率曲线
- 绘制收益率 Histogram
- 比较多只股票的波动大小
- 将示例股票替换成自己关注的公司

### 笔记产出

- 一张收益率分布图
- OHLCV 字段解释
- 股票波动大小比较
- 对收益率分布长尾现象的理解

## 五、第三章：移动平均线策略

预计时间：30～45 分钟

### 核心知识

- 趋势与短期噪声
- 移动平均线的作用
- MA5、MA20 的计算
- 金叉与死叉
- 将交易想法转换为代码规则

### 关键代码

- rolling(window).mean()
- signal = MA5 &gt; MA20
- 买入和卖出信号判断
- 均线与交易点可视化

### 实操任务

- 计算 MA5 和 MA20
- 标记金叉和死叉
- 将股票替换为 TSLA
- 比较 MA5/MA20 与 MA10/MA30
- 观察参数变化对交易次数的影响

### 笔记产出

- 一张策略信号图
- 双均线策略规则说明
- 两组均线参数的比较
- 对“金叉是否一定赚钱”的回答

## 六、第四章：策略回测

预计时间：30～45 分钟

### 核心知识

- 回测的定义与局限
- 持仓状态和策略收益
- 使用 shift(1) 避免未来函数
- 策略与买入持有基准
- 净值曲线、胜率和最大回撤

### 关键代码

- position.shift(1)
- 收益率累乘
- cummax()
- 最大回撤计算
- 策略净值与基准净值比较

### 实操任务

- 回测双均线策略
- 对比策略、买入持有和大盘
- 将标的替换为 NVDA
- 比较 PERIOD = "1y" 和 PERIOD = "5y"
- 分析不同周期的最大回撤

### 笔记产出

- 一张策略净值曲线
- 策略是否跑赢基准的结论
- 1 年与 5 年回测结果比较
- 用三句话解释什么是回测

## 七、Phase 1 成果清单

- [ ] 股票价格与成交量图
- [ ] 收益率分布图
- [ ] 移动平均线策略信号图
- [ ] 策略与基准净值曲线
- [ ] 四章挑战题答案
- [ ] Phase 1 学习总结

## 八、通关检查

- [ ] 能解释量化交易与主观交易的区别
- [ ] 能解释 OHLCV 每个字段
- [ ] 能使用 pct_change() 计算收益率
- [ ] 能使用 rolling() 计算移动平均线
- [ ] 能把交易想法转换为信号
- [ ] 知道为什么回测需要 shift(1)
- [ ] 能解释净值、胜率和最大回撤
- [ ] 知道历史回测不代表未来收益

## 九、阶段复盘

1. Phase 1 中最重要的三个概念是什么？
2. 哪部分代码最难理解？
3. 均线参数变化会怎样影响交易结果？
4. 当前回测忽略了哪些现实因素？
5. 下一阶段最想深入学习什么？

