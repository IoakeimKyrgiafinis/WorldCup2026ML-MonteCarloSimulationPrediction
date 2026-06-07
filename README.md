# WorldCup2026ML-MonteCarloSimulationPrediction

This project combines squad market valuations, historical match results, age factors, bookmaker odds, and environmental factors (heat stress, pressing intensity) to simulate the 2026 FIFA World Cup 10,000 times and estimate each team's probability of winning the tournament. The approach is inspired by Roll et al. (2019) and Zeileis et al. (2026). 

In short, a Random Forest Classifier is trained on an 80/20 train/test split. Train dataset features are extracted from publicly available Kaggle Datasets. The features extracted are the following: 

| Feature | What It Measures |
| :--- | :--- |
| `total_value_diff` | Squad market value gap (€) |
| `avg_value_diff` | Average player quality gap |
| `gini_diff` | Value distribution inequality gap |
| `form_diff` | Recent win rate gap (last 5 games) |
| `prime_diff` | Age-prime score gap |


*We compute the prime age score of each player in the following way using Gaussian Peak curve ​

$$\text{prime score}(age) = e^{-\frac{(age - 25.5)^2}{2 \times 3.5^2}}$$

where 25.5 is the peak age and 3.5 is the sigma (spread). (Branquinho et al., 2025)

The Random Forest Classifier model is trained on these features on club matches from 2005 onwards, not international matches. This is done because there are far more club matches available than international matches, which happen infrequently. The assumption is that football outcomes are affected in the same way by the features for both clubs and national teams. 

The target variable (`result`) takes three values: `1` (home win), `0` (draw), and `-1` (away win). Once trained, the model outputs three class probabilities for each result: $P(\text{home win})$, $P(\text{draw})$, and $P(\text{away win})$.

---

## Match Probability Engine

For the 2026 World Cup, the raw model output is not used directly. It goes through three sequential adjustments before becoming the final match probability.

### 1. Heat Stress Adjustment ($\alpha = 0.002$)
There is a lot of discussion about how the high temperatures present mainly in USA venues in June-July are going to affect player performance in the tournament. We try to account for it by first implementing a heat stress adjustment. 

* Each team has a baseline temperature (`team_baseline_temp`) representing their typical training climate. 
* Each venue has an expected match-day temperature (`venue_heat`). 

Heat stress is the gap between the venue temperature and the team baseline—a Senegalese team playing in Houston in July experiences very little additional stress compared to a Norwegian team. The heat difference between the two teams is computed and applied as a small adjustment to win probability ($\alpha = 0.002$ per degree Celsius of differential). The adjustment is clamped between `0.01` and `0.99` so probabilities never reach zero or absolute certainty. 

#### Extraction of $\alpha = 0.002$:
Mohr et al. (2012) observed that a massive temperature swing from a temperate baseline ($\sim 21^\circ\text{C}$) to extreme tournament heat ($\sim 43^\circ\text{C}$) caused a 7% total performance drop for unacclimatized players. The delta between those two test environments was exactly $22^\circ\text{C}$ ($43 - 21 = 22$). Taking that 7% total drop ($0.07$) and dividing it across that temperature gap yields:

$$\frac{0.07 \text{ total performance deficit}}{22^\circ\text{C} \text{ temperature delta}} \approx \mathbf{0.0031}$$

Accounting for modern acclimatization techniques, this scaling factor is smoothed down to a baseline of `0.002` per degree Celsius.

### 2. Tactical Pressing Penalty ($\alpha = 0.003$)
Next, we account for the fact that the participating teams have different playstyles. We extract each team's pressing intensity from analyst reports and penalize the high-pressure teams, based on the argument that heat is going to affect them at a higher rate. In extreme heat, a high-pressing team faces a double penalty: their tactical style becomes harder to maintain. The pressing adjustment models this interaction.

Each team has a `pressing_intensity` score (`0` to `1`). The adjustment scales the pressing differential by heat severity (venue temperature divided by 40, normalized). 

$$\text{Tactical Nudge} = 0.003 \times \Delta\text{Pressing Intensity} \times \text{Heat Severity}$$

* **Example:** A high-pressing team (Austria, 0.95) playing a low-pressing team (Qatar, 0.10) in Dallas ($38.5^\circ\text{C}$) would have their win probability nudged down by approximately $0.003 \times 0.85 \times 0.96 = 0.0024$—small but meaningful across the tournament bracket.

#### Extraction of $\alpha = 0.003$:
A high-pressing system relies entirely on sustaining continuous high-intensity running to choke the opponent's space. Conversely, a low-pressing system saves physical energy by sitting in a passive shape and focusing on possession mechanics. The paper proves that extreme heat creates a severe tactical disadvantage for high-intensity movement while rewarding a slower, cleaner passing style. 

To extract the mathematical "exchange rate" of this tactical trade-off, we evaluate the friction between physical decay and technical gains recorded by Mohr et al. (2012), dividing the passing efficiency gain (+8%) by the high-intensity running loss (-26%): 

$$\text{Tactical Exchange Rate} = \frac{\text{Passing Success Gain}}{\text{High-Intensity Running Loss}}$$

$$\text{Tactical Exchange Rate} = \frac{0.08}{0.26} \approx \mathbf{0.3076}$$

Shifting the decimal two places to the left to scale it down safely from a raw physical efficiency metric into a percentage-point modifier for a probability outcome loop ($0.3076 \times 0.01$) yields exactly **0.003** when rounded.

### 3. Bookmaker Odds Blending
After heat and pressing adjustments, the model probabilities are blended with bookmaker odds. This is done based on the argument that bookmaker odds encapsulate an enormous amount of information that a Machine Learning model trained purely on historical data cannot capture (squad news, injury reports, tactical adjustments, etc.). 

American odds are converted to implied probabilities using the standard formula, then normalized to sum to 1 across all teams. For each match, the relative winner odds of the two teams determine the odds-implied head-to-head win probability.

The final blended probability uses `odds_weight = 0.6`:
* **60% weight** to the bookmaker-implied probability
* **40% weight** to the model probability (after heat and pressing adjustments)

The draw probability uses the model's draw estimate as its odds anchor (since outright tournament winner odds don't price individual match draws), then blends with the same 40/60 split. All three probabilities are renormalized to sum to 1 after blending.

---

## Expected Goals ($\lambda$)

For every match, expected goals ($\lambda$) are computed for each team from their squad market value differential:
* $\lambda_a = \max(0.5, 1.5 + \frac{\text{value\_diff}}{1e9})$
* $\lambda_b = \max(0.5, 1.5 - \frac{\text{value\_diff}}{1e9})$

The baseline of 1.5 represents an average international match goal rate. The value differential shifts this—a €500M squad advantage adds 0.5 expected goals. The floor of 0.5 ensures no team's expected goals collapse to an unrealistic level.

Goals are sampled from a Poisson distribution—the standard model for discrete count data like football scores. Crucially, **rejection sampling** is used rather than clamping. 

The naive approach (sampling goals, then forcing the winner to have more by subtracting 1 from the loser) distorts the distribution, creating an artificial pile-up at scorelines like 1-0, 2-1, 3-2. Rejection sampling instead draws two independent Poisson samples and accepts them only if they are consistent with the simulated match outcome. With realistic lambdas, this converges in very few tries. If the sampler fails to converge within 500 attempts (extremely rare), a minimal fallback score is used (1-0, 0-1, or 0-0 for the respective outcome).

---

## Tournament Simulation

### Group Stage
Each of the 12 groups plays a full round-robin: every team faces every other team once (6 matches per group). For each match, the outcome (home win / draw / away win) is drawn from the cached probabilities, and a Poisson score is generated. Points (3/1/0), goal difference, and goals for are all accumulated.

Final group standings are sorted by points, goal difference, and goals for—exactly the FIFA tiebreaker order. The top two teams advance as group winner and runner-up. The third-place team's record is saved for the best-third-place ranking.

### Best Third-Place Teams
In a 48-team World Cup with 12 groups of 4, 8 third-place teams also advance to the Round of 32. The 12 third-place finishers are ranked by the same criteria (points, goal difference, goals for) and the top 8 advance. These are stored as `best8` and slotted into the bracket in the official FIFA-specified positions.

### Knockout Rounds
From the Round of 32 onwards, all matches are single-elimination. The bracket is hard-coded to match the official FIFA 2026 World Cup bracket structure, with each match numbered 73-104 and assigned to its official venue. For knockout matches, a draw in 90 minutes leads to a 50/50 penalty shootout coin flip. This is a simplification—in reality, the stronger team has a slight penalty advantage—but it is a reasonable approximation since penalty shootouts are largely unpredictable.

### Monte Carlo Engine Execution
The full tournament simulation is run 10,000 times. Each run is independent—group draws, scores, and knockout results are all re-sampled from scratch. The only shared state is the `matchup_cache` (pre-computed probabilities), which is deterministic and identical across all runs.

After 10,000 simulations, each team's win count is divided by 10,000 to produce a win probability percentage. The results are sorted from highest to lowest probability. 10,000 iterations is sufficient for stable probability estimates at the top of the table ($\pm0.5\%$ for teams with $10\%+$ win probability). For very low-probability teams (below 1%), more simulations would reduce noise further, but the absolute differences at that level are not practically meaningful.

---

## Limitations

* **Training Environment Disconnect:** Training on club data to predict international matches means the feature space is shared, but the context differs (squad size, player familiarity, tactical system cohesion).
* **Static Squad States:** No live injury or suspension modeling is integrated; a key player being suspended or injured for a knockout stage match cannot be captured.
* **Deterministic Shootouts:** Penalty shootouts are simulated as a static 50/50 coin flip, which ignores proven team and goalkeeper performance metrics during spot-kicks.
* **Simplified Seeding Rules:** The model places the `best8` third-place teams into bracket slots strictly by their ranking order, whereas FIFA's official seeding matrix uses more complex, group-dependent path-blocking constraints.
* **Outright to Match Probability Conversions:** Converting tournament outright winner odds to localized head-to-head match probabilities assumes that relative outright odds accurately approximate isolated match-level win distributions.

---

## Backtest Results

Below are the historical backtesting accuracy results evaluated across tournament configurations to measure the optimization added by combining market metrics with real-time betting lines.

Below are the aggregated title-winning probabilities derived from 10,000 Monte Carlo tournament iterations, comparing the standalone Machine Learning engine against the market-blended engine.

| Rank | Country | Sim Wins (Pure ML) | Win Probability (Pure ML) | Sim Wins (Blended 0.6) | Win Probability (Blended 0.6) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Brazil | 2929 | 29.29% | 2647 | 26.47% |
| 2 | England | 1789 | 17.89% | 1508 | 15.08% |
| 3 | France | 1051 | 10.51% | 1222 | 12.22% |
| 4 | Argentina | 382 | 3.82% | 1101 | 11.01% |
| 5 | Spain | 856 | 8.56% | 998 | 9.98% |
| 6 | Germany | 888 | 8.88% | 875 | 8.75% |
| 7 | Portugal | 1029 | 10.29% | 750 | 7.50% |
| 8 | Belgium | 197 | 1.97% | 304 | 3.04% |
| 9 | Netherlands | 145 | 1.45% | 294 | 2.94% |
| 10 | Denmark | 50 | 0.50% | 79 | 0.79% |
| 11 | Uruguay | 60 | 0.60% | 52 | 0.52% |
| 12 | Croatia | 82 | 0.82% | 32 | 0.32% |
| 13 | Serbia | 51 | 0.51% | 25 | 0.25% |
| 14 | United States | 36 | 0.30% | 17 | 0.17% |
| 15 | Senegal | 50 | 0.50% | 15 | 0.15% |
| 16 | Wales | -- | -- | 11 | 0.11% |
| 17 | Morocco | 83 | 0.83% | 9 | 0.09% |
| 18 | Canada | 65 | 0.65% | 9 | 0.09% |
| 19 | Switzerland | -- | -- | 8 | 0.08% |
| 20 | Mexico | 30 | 0.30% | 8 | 0.08% |

The model does a good job in ranking team probabilities according to the bookmakers odds favourites at that specific time. Still it does not correctly predict the winners Argentina since it is trained on features related to squad value, an aspect in which Argentina was not the dominant team.

---

## Reference List

* Mohr, M., Nybo, L., Grantham, J., & Racinais, S. (2012). Physiological responses and physical performance during football in the heat. *PLoS ONE*, 7(6), e39202. https://doi.org/10.1371/journal.pone.0039202

* Branquinho, L., de França, E., Titton, A., Leite de Barros, L. F., Campos, P., Marques, F. O., Glória, I. P. dos S., Caperuto, E. C., Hirota, V. B., Teixeira, J. E., Forte, P., Monteiro, A. M., Ferraz, R., & Thomatieli-Santos, R. V. (2025). The aging curve: How age affects physical performance in elite football. Journal of Functional Morphology and Kinesiology, 10(4), 385. https://doi.org/10.3390/jfmk10040385
