# 组合账户统一 + 组合级风控（Portfolio Account & Risk Layer）

- 日期：2026-10-05
- 状态：Final（已定稿）
- 范围：`live-strategy`
- 部署环境：AWS EC2（本机 `logs/`、`state/` 与生产无关，仅作本地开发副本）

---

## 1. 背景与问题

当前策略把资金记在**每个 pair 上**：

- `PairState.cash` 默认 `100_000`（`state.py:16`）。
- 每个 pair 独立 `PairState.load(p)`（`run.py:67`），文件不存在就各自拿默认 10 万。
- `compute_equity(state, ...)` 是 per-pair 的（`run.py:438-444`）：`state.cash + 该 pair 持仓价值`，无跨 pair 汇总。
- 回测同样单 pair 各跑一遍（`backtest.py:108`，`init_cash` 各 10 万）。
- `exchange_client.get_balance()`（`exchange_client.py:94`）从未被 `run.py` 调用。

核心问题：

1. **资金不共享**：加标的 = 每个新 pair 再开一本 10 万账，本地以为总资金是 `N × 100k`，
   而比赛真实账户只有 10 万。sizing 的分母（equity）因此被系统性高估。
2. **记账会漂**：本地现金/持仓与 Roostoo 真实余额脱节（拒单、部分成交、手续费取整等），
   需要每 loop 用 `get_balance()` 把现金和持仓拉回真实值。

## 2. 目标 / 非目标

### 目标

- **G1 单一共享账户**：全组合唯一 `cash` / `equity`，不再 per-pair。
- **G2 每-loop 对账（现金 + 持仓）**：`get_balance()` 是现金与持仓数量的唯一真相来源，
  每 loop 覆盖本地记账，自愈漂移、抗 EC2 重启/异常。
  `equity = 对账现金 + Σ 持仓价值`。
- **G3 组合级风控（与 per-asset 不重叠）**：
  - 全局总敞口上限（Σ 持仓 notional ≤ X% 组合 equity）。
  - 全局回撤熔断（回撤 > X% 停开新仓，回落到 X/2 恢复；**绝不减仓现有持仓**）。
  - 单资产上限改为相对**组合 equity**（复用现有 `MAX_POSITION_PCT`，只换分母）。
- **G4 组合级回测**：多资产共享资金一起回测，输出组合级 equity/回撤/Sharpe，用于定阈值。

### 非目标（明确不做）

- 不做**再平衡**（主动修剪超重资产；会产生手续费，另议）。
- 不做 **limit 单入场/出场**（另立 spike 探索）。
- 不**扩展更多币对**（数据源问题另议）。
- 不**改 per-asset 退出逻辑**（stop / TP / MA-cross 保持原样）。

> 关键约束（来自上一轮结论）：组合层**只做 per-asset 逻辑做不了的事**，且**永不卖出已有持仓**，
> 以避免与 per-asset 的 stop/TP/MA-cross 重合、产生冗余交易和多余手续费。

## 3. 现状定位（改动前的事实）

| 关注点 | 位置 | 现状 |
|---|---|---|
| per-pair 现金 | `state.py:16` | `cash: float = 100_000.0` |
| pair 独立加载 | `run.py:67` | `states = {p: PairState.load(p) for p in PAIRS}` |
| per-pair equity | `run.py:438-444` | 只含本 pair 的 cash + 持仓 |
| sizing 分母 | `run.py:223-229`、`run.py:361-367` | 传 per-pair `equity`/`cash` |
| 真实余额 | `exchange_client.py:94` | `get_balance()` 存在，未被 `run.py` 调用 |
| 回测 | `backtest.py:108` | 单 pair、`init_cash=INITIAL_CASH` |
| 两个 10 万来源 | `config.py:22` vs `state.py:16` | `INITIAL_CASH`（backtest 用）与 `PairState.cash` 默认值（live 用）互不对齐 |

## 4. 设计

### 4.1 状态模型

**新增 `PortfolioState`（全组合唯一账户）**，放在 `state.py`，持久化到新文件
`logs/state/state_PORTFOLIO.json`：

```python
@dataclass
class PortfolioState:
    cash: float = INITIAL_CASH          # 唯一现金，每 loop 从 get_balance() 对账
    peak_equity: float = INITIAL_CASH   # 历史峰值（用于回撤熔断）
    breaker_active: bool = False        # 熔断是否已触发
    # save()/load() 复用 PairState 的 JSON 持久化方式
```

**`PairState` 删除 `cash` 字段**（`state.py:16`），只保留持仓与决策元数据：

- 多头：`position`、`entry_price`、`stop_price`、`stop_armed`、`long_atr_at_entry`、
  `long_original_qty`、`long_tp1_done`、`long_tp2_done`
- 空头：`short_position`、`short_entry_price`、`short_stop_price`、`short_stop_armed`、
  `short_collateral`、`short_original_qty`、`short_atr_at_entry`、`short_tp1_done`、`short_tp2_done`
- 共享：`prev_short_ma`、`prev_long_ma`、`last_decision_ts`

> 原则：**现金归 Portfolio；持仓数量每 loop 从钱包对账；entry_price/TP/止损等决策元数据由策略本地维护。**

### 4.2 equity 计算与对账（G2）

每 loop 开始时执行对账，`get_balance()` 是现金与持仓数量的唯一真相来源：

```
cash          = SpotWallet["USD"]["Free"]
position_i    = SpotWallet[coin_i]["Free"]      # BTC/USD -> "BTC"，ETH/USD -> "ETH"
equity        = cash
              + Σ_i( position_i × price_i )      # price_i 用 Roostoo ticker
              + short_unrealized_pnl              # 来自 /v6/short_positions
```

对账动作：

1. `portfolio.cash = SpotWallet["USD"]["Free"]`（覆盖，不累加）。
2. 每个 pair：`PairState.position = SpotWallet[coin_i]["Free"]`；空头持仓从
   `/v6/short_positions` 覆盖。
3. `entry_price`、TP/止损标志等决策元数据仍由策略自己在成交时写入（钱包不提供这些字段）。

这样 bot 在 EC2 上重启、被拒单、部分成交后，下一个 loop 自动拉回真实现金与持仓，实现自愈。

### 4.3 资金/持仓路由（G1）

`run.py` 中所有 `state.cash` 读写改为 `portfolio.cash`（全组合唯一），完整清单：

| 位置 | 现写法 | 改法 |
|---|---|---|
| 长仓入场条件 `run.py:221` | `state.cash > 0` | `portfolio.cash > 0` |
| 长仓开仓扣款 `run.py:242` | `state.cash -= filled*fill + fee` | `portfolio.cash -= ...` |
| 长仓止盈/退出回款 `run.py:169`、`run.py:199` | `state.cash += ...` | `portfolio.cash += ...` |
| 空仓入场条件 `run.py:359` | `state.cash > 0` | `portfolio.cash > 0` |
| 空仓 collateral 上限 `run.py:367` | `min(collateral, state.cash)` | `min(collateral, portfolio.cash)` |
| 空仓开仓锁 collateral `run.py:378` | `state.cash -= collateral + fee` | `portfolio.cash -= ...` |
| 空仓平仓回款 `run.py:455` | `state.cash += return_amount` | `portfolio.cash += ...` |
| sizing 调用 `run.py:223-229`、`run.py:361-367` | 传 per-pair `equity`/`cash` | 传组合 `equity`/`cash` |
| `compute_equity` `run.py:438-444` | per-pair | 组合级（见 4.2） |

> `portfolio.cash` 是 loop 内的运行值：loop 开头从 `get_balance()` 对账，随后订单先后成交时
> 递减/递增；下个 loop 开头再次被 `get_balance()` 拉回真实值，不累积长期漂移。

### 4.4 组合级风控（G3，与 per-asset 不重叠）

新增 `PortfolioManager`（建议新文件 `portfolio.py`），只做三件事，**永不主动平仓**：

**① 全局总敞口上限**

```
gross = Σ_i( pair_i.position × price_i ) + Σ_j( short_notional_j )
if gross + planned_notional > MAX_TOTAL_EXPOSURE × equity:
    planned_notional = max(0, MAX_TOTAL_EXPOSURE × equity - gross)   # 缩量到剩余额度
    if planned_notional < pair 的最小下单量: 跳过入场
```

**② 全局回撤熔断（只停新仓）**

```
drawdown = (peak_equity - equity) / peak_equity
if not active and drawdown >= DRAWDOWN_TRIGGER:  active = True   # 停开新仓
if active     and drawdown <= DRAWDOWN_RECOVER:  active = False  # 恢复开仓
peak_equity = max(peak_equity, equity)                            # 每 loop 更新并持久化
```

`active` 期间：跳过所有 pair 的**入场块**（long entry / short entry）；退出/TP/止损照常运行。

**③ 单资产上限（换分母，不加新机制）**

现有 `MAX_POSITION_PCT` 语义不变，仅把分母从 per-pair equity 换成组合 equity ——
通过 4.3 里把组合 equity 传入 `compute_position_size`/`compute_short_size` 自动达成。
`strategy.py` 无需改动（两个 sizing 函数已是参数化）。

> 明确**删除**上一版中的「回撤按比例减仓」「暴露超限事后强制减仓」——它们与 per-asset
> stop/TP 重合，会造成冗余交易。

### 4.5 回测改造（G4）

`backtest.py` 从「单 pair 各跑一遍」改为「多资产共享资金一起跑」：

- 输入：多个 OHLCV CSV，按时间戳对齐。
- 账户：单个 `PortfolioState`（共享 `cash`/`peak_equity`）。
- 逐 bar：算组合 equity → 熔断判定 → 每个 pair 出信号 → 入场受总敞口上限约束 → 退出照常。
- 输出：**组合级** equity curve、max drawdown、Sharpe/Sortino/Calmar、`composite_score`。

这是 G3 阈值的唯一验证手段，必须先于 live 上线。

### 4.6 配置参数（`config.py`）

```python
INITIAL_CASH        = 100_000.0   # 组合唯一初始资金（回测用；live 首 loop 被 get_balance 覆盖）
MAX_TOTAL_EXPOSURE  = 0.60        # 全局总敞口 ≤ 60% 组合 equity（默认，回测定）
MAX_POSITION_PCT    = 0.20        # 单资产 ≤ 20% 组合 equity（语义不变，分母改）
DRAWDOWN_TRIGGER    = 0.10        # 回撤熔断触发线（默认，回测定）
DRAWDOWN_RECOVER    = 0.05        # 回撤熔断恢复线（默认，回测定）
```

### 4.7 文件改动清单

| 文件 | 改动 |
|---|---|
| `state.py` | 新增 `PortfolioState`；`PairState` 删 `cash`；`save/load` 适配 |
| `portfolio.py`（新） | `PortfolioManager`：现金/持仓对账、equity 聚合、总敞口上限、回撤熔断 |
| `run.py` | 初始化 `portfolio` + 每 loop 对账；`state.cash` → `portfolio.cash`；入场加敞口/熔断门；`compute_equity` 组合化 |
| `backtest.py` | 多资产共享资金回测引擎 |
| `config.py` | 新参数；`INITIAL_CASH` 语义统一 |
| `strategy.py` | **无改动**（sizing 已参数化） |

## 5. 部署/迁移（EC2）

- bot 跑在 EC2，state 持久化在 EC2 本地文件系统。新版本上线：
  1. 旧 per-pair state 文件（含 `cash`）作废；`PairState.load()` 忽略未知字段即可（现有逻辑已兼容）。
  2. **无需手工迁移**：首个 loop 的 `get_balance()` 对账即把现金与持仓设为真实值。
- 上线顺序：先在 `DRY_RUN` 验证对账与门控逻辑 → 再用组合级回测定阈值 → 最后切 live。

## 6. 测试与验证

1. **组合级回测**：用历史数据跑 BTC+ETH 共享资金，对比改动前后的组合级 max drawdown/Sharpe。
2. **对账正确性**：`DRY_RUN` 下确认每 loop 后 `portfolio.cash` 与 `get_balance()` 的 USD 一致、
   持仓与钱包一致（现金与持仓都被钱包覆盖）。
3. **熔断/敞口单测**：构造 equity 曲线触发/恢复熔断，确认「只停新仓、不平仓」。
4. **回归**：per-asset 的 stop/TP/MA-cross 行为与改动前一致（不因账户统一而变化）。

## 7. 定稿决策

1. **对账范围**：现金 + 持仓均每 loop 从 `get_balance()` 覆盖（自愈）。
2. **阈值**：`MAX_TOTAL_EXPOSURE=0.60`、`DRAWDOWN_TRIGGER=0.10`、`DRAWDOWN_RECOVER=0.05`、
   `MAX_POSITION_PCT=0.20`（默认值，最终以组合级回测结果为准）。
3. **`peak_equity` 起点**：首次运行以当前 equity 为峰值起点（避免历史回撤误触发）。
