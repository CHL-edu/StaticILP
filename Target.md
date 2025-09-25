论文《A novel integer linear programming model for routing and spectrum assignment in optical networks》：

---

# 论文算法实现：弹性光网络中的路由与频谱分配（RSA）问题的完整边-节点ILP模型

## 1. 问题定义 (Problem Definition)

**输入 (Input):**
*   **光谱 `S`**: `S = {1, 2, ..., s̄}`，表示可用的连续频隙集合。
*   **光网络 `G=(V, E)`**: 一个无向、无自环、连通图。`V` 是节点集，`E` 是边（光纤链路）集。每条边 `e ∈ E` 有一个长度 `lₑ`。
*   **需求集合 `K`**: 每个需求 `k ∈ K` 包含：
    *   源节点 `oₖ ∈ V` 和目的节点 `dₖ ∈ V \ {oₖ}`。
    *   请求的频隙数量 `wₖ ∈ N⁺`。
    *   传输距离限制 `l̄ₖ ∈ R⁺`。

**输出 (Output):**
为每个被接受的需求 `k` 确定一个**光路**（lightpath），包含：
1.  **路由 `Pₖ`**: 一条从 `oₖ` 到 `dₖ` 的路径，满足 `∑ₑ∈Pₖ lₑ ≤ l̄ₖ`。
2.  **频谱分配 `Sₖ`**: 一组 `wₖ` 个**连续**的频隙，分配给路径 `Pₖ` 上的**所有**边，且与其他需求在共享边上的频谱不重叠。

**核心约束 (Core Constraints):**
*   **频谱连续性 (Continuity)**: 一个需求在路径所有边上分配的频隙必须相同。
*   **频谱邻接性 (Contiguity)**: 分配给一个需求的频隙必须是连续的。
*   **频谱非重叠 (Non-overlapping)**: 在任何一条边上，一个频隙只能分配给一个需求。

**优化目标 (Objective Functions):**
*   **O1**: 最小化所有路径的总跳数（边数）。
*   **O2**: 最小化用于路由的边的总数。
*   **O3**: 最小化使用的最大频隙位置。

---

## 2. 决策变量 (Decision Variables)

### 2.1 路由变量 (Routing Variables)
*   **`xₑₖ`** ∈ {0, 1}:
    *   `xₑₖ = 1` 表示需求 `k` 的路径经过边 `e`。
    *   `xₑₖ = 0` 表示未经过。

### 2.2 频谱分配变量 (Spectrum Assignment Variables)
*   **`zₛₖ`** ∈ {0, 1}:
    *   `zₛₖ = 1` 表示频隙 `s` 是分配给需求 `k` 的**最后一个**频隙（即占用 `[s-wₖ+1, s]`）。
*   **`tₑ,ₛ,ₖ`** ∈ {0, 1}:
    *   `tₑ,ₛ,ₖ = 1` 表示频隙 `s` 在边 `e` 上被分配给了需求 `k`。

### 2.3 辅助变量 (Auxiliary Variables)
*   **`aₑ`** ∈ {0, 1} (用于 O2):
    *   `aₑ = 1` 表示至少有一个需求的路径经过了边 `e`。
*   **`pₑ`** ∈ Z⁺ (用于 O3):
    *   `pₑ` 表示在边 `e` 上使用的最大频隙位置。
*   **`p`** ∈ Z⁺ (用于 O3):
    *   `p` 表示在整个网络中使用的最大频隙位置。

---

## 3. 约束条件 (Constraints)

### 3.1 路由约束 (Routing Constraints)
这些约束确保 `xₑₖ` 的取值能构成一条从 `oₖ` 到 `dₖ` 的有效路径，并满足传输距离限制。

*   **源点约束 (Origin):**
    ```math
    \sum_{e \in \delta(o_k)} x_{e,k} \leq 1, \quad \forall k \in K
    ```
*   **终点约束 (Destination):**
    ```math
    \sum_{e \in \delta(d_k)} x_{e,k} - \sum_{e \in \delta(o_k)} x_{e,k} = 0, \quad \forall k \in K
    ```
*   **路径连续性约束 (Path-Continuity) - 指数级数量:**
    ```math
    \sum_{e \in \delta(X)} x_{e,k} - \sum_{e \in \delta(o_k)} x_{e,k} \geq 0, \quad \forall k \in K, \forall X \subset V ,o_k \in X, d_k \in V \setminus X
    ```
*   **度数约束 (Degree Constraints):**
    ```math
    \sum_{e \in \delta(v)} x_{e,k} \leq 2, \quad \forall k \in K, \forall v \in V \setminus \{o_k, d_k\}
    ```
*   **环路消除约束 (Cycle-Elimination) - 指数级数量:**
    ```math
    \sum_{e' \in \delta(X_e)} x_{e',k} \geq 
    \begin{cases} 
    2x_{e,k} & \text{if } |X_e \cap \{o_k, d_k\}| = 0 \\
    x_{e,k} & \text{if } |X_e \cap \{o_k, d_k\}| = 1 
    \end{cases}, \quad \forall k \in K, \forall e \in E, \forall X_e \subset V 
    ```
*   **传输距离约束 (Transmission-Reach):**
    ```math
    \sum_{e \in E} l_e x_{e}^{k} - \bar{l}_k \sum_{e \in \delta(o_k)} x_{e}^{k} \leq 0, \quad \forall k \in K
    ```

### 3.2 边激活约束 (Edge Activation for O2)
*   ```math
    a_e - x_{e,k} \geq 0, \quad \forall k \in K, \forall e \in E
    ```
*   ```math
    a_e \leq \sum_{k \in K} x_{e,k}, \quad \forall e \in E
    ```

### 3.3 频谱分配约束 (Spectrum Assignment Constraints)
这些约束确保 `zₛₖ` 和 `tₑ,ₛ,ₖ` 能构成一个有效的频谱分配方案。

*   **通道选择约束 (Channel Selection):**
    ```math
    \sum_{s=w_k}^{\bar{s}} z_{s,k} - \sum_{e \in \delta(o_k)} x_{e,k} = 0, \quad \forall k \in K
    ```
*   **禁用频隙约束 (Forbidden-Slot):**
    ```math
    \sum_{s=1}^{w_k-1} z_{s,k} = 0, \quad \forall k \in K
    ```
*   **边-频隙约束 (Edge-Slot):**
    ```math
    \sum_{s \in S} t_{e,s}^{k} - w_k x_{e,k} = 0, \quad \forall k \in K, \forall e \in E
    ```
*   **频谱连续性与邻接性约束 (Continuity & Contiguity):**
    ```math
    x_{e}^{k} + \sum_{s'=s}^{\min(s+w_k-1, \bar{s})} z_{s'}^{k} - t_{e,s}^{k} \leq 1, \quad \forall k \in K, \forall e \in E, \forall s \in S
    ```
*   **频谱非重叠约束 (Non-overlapping):**
    ```math
    \sum_{k \in K} t_{e,s}^{k} \leq 1, \quad \forall e \in E, \forall s \in S
    ```

### 3.4 最大频隙位置约束 (Max-Slot Position for O3)
*   ```math
    s \cdot t_{e,s}^{k} - p_e \leq 0, \quad \forall k \in K, \forall e \in E, \forall s \in S
    ```
*   ```math
    p_e - \sum_{k \in K} \sum_{s \in S} s \cdot t_{e,s}^{k} \leq 0, \quad \forall e \in E
    ```
*   ```math
    p_e \leq p \leq \bar{s}, \quad \forall e \in E
    ```

---

## 4. 目标函数 (Objective Functions)

*   **O1 (最小化总跳数):**
    ```math
    \min \sum_{e \in E} \sum_{k \in K} x_{e,k}
    ```
*   **O2 (最小化使用的边数):**
    ```math
    \min \sum_{e \in E} a_e
    ```
*   **O3 (最小化最大频隙位置):**
    ```math
    \min p
    ```

---

## 5. 求解策略 (Solution Strategy)

由于模型包含**指数级数量**的路径连续性约束 (3) 和环路消除约束 (5)，直接求解不可行。论文采用 **分支切割法 **(Branch-and-Cut)：

1.  **初始化**: 从一个松弛模型开始，只包含多项式数量的约束（如源点、终点、度数、频谱等约束）。
2.  **求解LP松弛**: 使用ILP求解器（如CPLEX）求解当前模型的线性规划松弛。
3.  **分离 (Separation)**:
    *   对于每个需求 `k`，检查当前解是否违反了任何路径连续性约束 (3)。这可以转化为在图 `G` 上寻找一个最小割问题。
    *   对于每个需求 `k` 和每条边 `e`，检查当前解是否违反了任何环路消除约束 (5)。这可以转化为在一个辅助图上寻找一个最小割问题。
4.  **添加切割**: 如果找到违反的约束，则将其作为“用户切割”(user cut) 添加到模型中。
5.  **迭代**: 重复步骤 2-4，直到没有新的违反约束被找到。
6.  **分支**: 如果当前解不是整数解，则进行标准的分支定界过程。

**关键优势**: 路径连续性和环路消除约束的分离问题都可以在**多项式时间**内解决（例如，使用Goldberg-Tarjan的预流推进算法），这使得整个分支切割框架在计算上是可行的。

---
