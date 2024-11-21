import os
import json
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

class PoissonModel:
    def __init__(self, competition, max_games=380, ignored_games=list(), x0=None, decay=0):
        self.x0 = x0
        self.max_games = max_games
        self.competition = competition
        self.ignored_games = ignored_games
        self.decay = decay  # Fator de esquecimento (xi)
        # self.filename_tag = f'{self.competition}_{self.year}_poisson_{self.max_games}_games_4_pars'

    # Função de verossimilhança com fator de esquecimento (xi)
    def likelihood(self, parameters, played_games, inx, times):
        lik = 0
        for home in played_games:
            for away in played_games[home]:
                result = played_games[home][away]
                t = times[home][away]  # Tempo do jogo (diferença de datas)

                # Fator de esquecimento
                phi = np.exp(-self.decay * t)

                # Probabilidade de gols do time da casa
                inx_1, inx_2 = inx[home]['Atk_home'] - 1, inx[away]['Def_away'] - 1
                if inx_1 == -1: 
                    mu_home = parameters[inx_2]
                else: 
                    mu_home = parameters[inx_1] * parameters[inx_2]
                lik -= phi*poisson.logpmf(result[0], mu_home)

                # Probabilidade de gols do time visitante
                inx_1, inx_2 = inx[away]['Atk_away'] - 1, inx[home]['Def_home'] - 1
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
        times = dict()
        inx_count = 0
        bounds = list()

        for game in data:
            if int(game) > self.max_games or int(game) in self.ignored_games: continue
            game = str(game).zfill(3)
            home, away, result = data[game]['Home'], data[game]['Away'], data[game]['Result']
            time_diff = self.max_games - int(game)
            result = result.upper().split(' X ')
            result = [int(x) for x in result]

            # Registrar os jogos
            if home not in played_games: 
                played_games[home] = dict()
                times[home] = dict()
            played_games[home][away] = result
            times[home][away] = time_diff

            # Verificar e adicionar todos os 4 parâmetros para o time da casa (home)
            if home not in inx:
                inx[home] = {'Atk_home': inx_count, 'Def_home': inx_count + 1, 
                            'Atk_away': inx_count + 2, 'Def_away': inx_count + 3}
                bounds.extend([(0, None)] * 4)
                inx_count += 4

            # Verificar e adicionar todos os 4 parâmetros para o time visitante (away)
            if away not in inx:
                inx[away] = {'Atk_home': inx_count, 'Def_home': inx_count + 1, 
                            'Atk_away': inx_count + 2, 'Def_away': inx_count + 3}
                bounds.extend([(0, None)] * 4)
                inx_count += 4
        
        bounds.pop(0)  # Remover limite inferior para o primeiro parâmetro, que é fixo em 1
        return played_games, inx, bounds, times


    # Otimizar parâmetros com base nos jogos
    def optimize_parameters(self, verbose):
        played_games, inx, bounds, times = self.preprocessing()
        parameters = np.random.random(4 * len(inx) - 1) if self.x0 is None else self.x0

        # Otimização usando a verossimilhança
        res = minimize(self.likelihood, parameters, args=(played_games, inx, times), bounds=bounds)
        
        if not res.success and verbose:
            print(f'Os parâmetros não convergiram após as tentativas.')

        parameters = np.hstack([np.array([1]), res.x])
        
        for club in inx:
            for force in inx[club]:
                inx[club][force] = parameters[inx[club][force]]

        return res.success, inx, parameters

    # Previsão da probabilidade de um placar específico
    def predict_score_probability(self, home_team, away_team, inx, score_home, score_away):
        mu_home = inx[home_team]['Atk_home'] * inx[away_team]['Def_away']
        mu_away = inx[away_team]['Atk_away'] * inx[home_team]['Def_home']
        
        prob_home = poisson.pmf(score_home, mu_home)
        prob_away = poisson.pmf(score_away, mu_away)
        
        return prob_home * prob_away

    # Previsão da probabilidade de vitória, empate e derrota
    def predict_match_probability(self, home_team, away_team, inx, max_goals=10):
        home_win_prob = 0
        away_win_prob = 0
        draw_prob = 0

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                joint_prob = self.predict_score_probability(home_team, away_team, inx, home_goals, away_goals)

                if home_goals > away_goals:
                    home_win_prob += joint_prob
                elif home_goals < away_goals:
                    away_win_prob += joint_prob
                else:
                    draw_prob += joint_prob

        return home_win_prob, draw_prob, away_win_prob
    
    # Executar o modelo para um conjunto de jogos e prever placares
    def run_model(self, verbose=True):
        success, inx, parameters = self.optimize_parameters(verbose)
        if success and verbose:
            print(f'Parâmetros estimados com sucesso!')
        return parameters
