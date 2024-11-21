import os
import json
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

class PoissonModel:
    def __init__(self, competition, year, max_games=380, ignored_games=list(), x0=None, gamma=1, decay=0):
        self.x0 = x0
        self.year = year
        self.max_games = max_games
        self.competition = competition
        self.ignored_games = ignored_games
        self.gamma = gamma  # Fator casa inicial
        self.decay = decay  # Fator de esquecimento (xi)
        # self.filename_tag = f'{self.competition}_{self.year}_poisson_{self.max_games}_games_{40 + self.home_away_pars}_pars'

    # Função de verossimilhança com fator de esquecimento (xi) e fator casa (gamma)
    def likelihood(self, parameters, played_games, inx, times):
        lik = 0
        gamma = parameters[-1]  # O último parâmetro será o fator casa (gamma)
        
        for home in played_games:
            for away in played_games[home]:
                result = played_games[home][away]
                t = times[home][away]  # Tempo do jogo (diferença de datas)

                # Fator de esquecimento
                phi = np.exp(-self.decay * t)

                # Probabilidade de gols do time da casa
                inx_1, inx_2 = inx[home]['Atk'] - 1, inx[away]['Def'] - 1
                if inx_1 == -1: 
                    mu_home = parameters[inx_2]*gamma
                else: 
                    mu_home = parameters[inx_1] * parameters[inx_2]*gamma
                lik -= phi*poisson.logpmf(result[0], mu_home)

                # Probabilidade de gols do time visitante
                inx_1, inx_2 = inx[away]['Atk'] - 1, inx[home]['Def'] - 1
                if inx_1 == -1: 
                    mu_away = parameters[inx_2]
                else: 
                    mu_away = parameters[inx_1] * parameters[inx_2]
                lik -= phi*poisson.logpmf(result[1], mu_away)

        return lik

    # Carregar e processar dados
    def preprocessing(self):
        with open(f'{self.competition}', 'r') as f:
            data = json.load(f)

        inx = dict()
        played_games = dict()
        times = dict()  # Para armazenar os tempos de cada jogo (para o fator de esquecimento)
        inx_count = 0
        bounds = list()

        for game in data:
            if int(game) > self.max_games or int(game) in self.ignored_games: continue
            game = str(game).zfill(3)
            home, away, result = data[game]['Home'], data[game]['Away'], data[game]['Result']
            time_diff = self.max_games - int(game)  # Diferença de tempo para o fator de esquecimento
            result = result.upper().split(' X ')
            result = [int(x) for x in result]

            # Registrar os jogos
            if home not in played_games: 
                played_games[home] = dict()
                times[home] = dict()
            played_games[home][away] = result
            times[home][away] = time_diff  # Adicionar o tempo do jogo

            # Índices para ataque e defesa do time da casa (home)
            if home not in inx:
                inx[home] = dict()
                inx[home]['Atk'] = inx_count
                bounds.append((0, None))
                inx_count += 1
                inx[home]['Def'] = inx_count
                bounds.append((0, None))
                inx_count += 1

            # Índices para ataque e defesa do time visitante (away)
            if away not in inx:
                inx[away] = dict()
                inx[away]['Atk'] = inx_count
                bounds.append((0, None))
                inx_count += 1
                inx[away]['Def'] = inx_count
                bounds.append((0, None))
                inx_count += 1

        # bounds.pop(0)  # Remover limite inferior para o primeiro parâmetro, que é fixo em 1
        return played_games, inx, bounds, times

    # Otimizar parâmetros com base nos jogos
    def optimize_parameters(self, verbose):
        played_games, inx, bounds, times = self.preprocessing()
        parameters = np.random.random(2 * len(inx)) if self.x0 is None else self.x0
        
        # Otimização usando a verossimilhança
        res = minimize(self.likelihood, parameters, args=(played_games, inx, times), bounds=bounds)
        
        if not res.success and verbose:
            print(f'Os parâmetros não convergiram após as tentativas.')

        # Atualiza os parâmetros otimizados
        parameters = np.hstack([np.array([1]), res.x[:-1]])  # O primeiro ataque é fixo em 1, gamma e xi são os últimos parâmetros
        self.gamma = res.x[-1]  # Atualiza o fator casa (gamma)
        
        for club in inx:
            for force in inx[club]:
                inx[club][force] = parameters[inx[club][force]]

        return res.success, inx, self.gamma, parameters
    
    # Previsão da probabilidade de um placar específico
    def predict_score_probability(self, home_team, away_team, inx, score_home, score_away, gamma):
        # Cálculo das médias esperadas de gols (mu) com o fator casa para o time da casa
        mu_home = inx[home_team]['Atk'] * inx[away_team]['Def'] * gamma
        mu_away = inx[away_team]['Atk'] * inx[home_team]['Def']
        
        # Probabilidade de o time da casa marcar 'score_home' gols e o visitante marcar 'score_away' gols
        prob_home = poisson.pmf(score_home, mu_home)
        prob_away = poisson.pmf(score_away, mu_away)
        
        # Retorna a probabilidade conjunta
        return prob_home * prob_away

    # Previsão da probabilidade de vitória, empate e derrota
    def predict_match_probability(self, home_team, away_team, inx, gamma, max_goals=10):
        # Inicializar as probabilidades de vitória, empate e derrota
        home_win_prob = 0
        away_win_prob = 0
        draw_prob = 0

        # Calcular a probabilidade para cada combinação de gols até 'max_goals'
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                # Usar o método predict_score_probability para calcular a probabilidade conjunta
                joint_prob = self.predict_score_probability(home_team, away_team, inx, home_goals, away_goals, gamma)

                # Somar a probabilidade para vitória da casa, empate ou vitória visitante
                if home_goals > away_goals:
                    home_win_prob += joint_prob
                elif home_goals < away_goals:
                    away_win_prob += joint_prob
                else:
                    draw_prob += joint_prob

        return home_win_prob, draw_prob, away_win_prob
    

      # Executar o modelo para um conjunto de jogos e prever placares
    def run_model(self, verbose=True):
        success, inx, gamma, parameters = self.optimize_parameters(verbose)
        if success and verbose:
            print(f'Parâmetros estimados com sucesso!')
        return parameters