#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
html parse code - MLB
@author: brianszekely
"""
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
from numpy import nan
from time import sleep
from os.path import join, exists
from os import getcwd
from urllib import request
from urllib.request import Request, urlopen
from pandas import read_csv
from numpy import where
from re import search
from difflib import get_close_matches

def get_data_team(team,year):
    #example url: https://www.baseball-reference.com/teams/tgl.cgi?team=NYY&t=b&year=2022
    #Batting URL
    URL_b = f'https://www.baseball-reference.com/teams/tgl.cgi?team={team}&t=b&year={year}'
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    req_1 = Request(URL_b,headers=hdr)
    html_1 = request.urlopen(req_1)
    soup_bat = BeautifulSoup(html_1, "html.parser")
    sleep(3)
    #Pitching
    URL_p = f'https://www.baseball-reference.com/teams/tgl.cgi?team={team}&t=p&year={year}'
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    req_1 = Request(URL_p,headers=hdr)
    html_1 = request.urlopen(req_1)
    soup_pitch = BeautifulSoup(html_1, "html.parser")
    sleep(3)
    #Game-by-Game schedule
    URL_sched = f'https://www.baseball-reference.com/teams/{team}/{year}-schedule-scores.shtml'
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}
    req_1 = Request(URL_sched,headers=hdr)
    html_1 = request.urlopen(req_1)
    soup_sched = BeautifulSoup(html_1, "html.parser")

    #Batting data
    game_result = []
    PA = []
    AB = []
    R = []
    H = []
    second_base = []
    third_base = []
    HR = []
    RBI = []
    BB = []
    IBB = []
    SO = []
    HBP = []
    SH = []
    SF = []
    ROE = []
    GIDP = []
    SB = []
    CS = []
    batting_avg = []
    onbase_perc = []
    slugging_perc = []
    onbase_plus_slugging = []
    LOB = []
    batters_number = []
    team_homeORaway = []
    table = soup_bat.find(id='all_team_batting_gamelogs')
    tbody = table.find('tbody')
    tr_body = tbody.find_all('tr')
    for trb in tr_body:
        for td in trb.find_all('td'):
            if td.get('data-stat') == "game_result":
                if 'W' in td.get_text():
                    game_result.append(1)
                else:
                    game_result.append(0)
            if td.get('data-stat') == "team_homeORaway":
                #Away = 0, Home = 1
                if '@' in td.get_text():
                    team_homeORaway.append(0)
                else:
                    team_homeORaway.append(1)
            if td.get('data-stat') == 'PA':
                PA.append(td.get_text())
            if td.get('data-stat') == 'AB':
                AB.append(td.get_text())
            if td.get('data-stat') == 'R':
                R.append(td.get_text())
            if td.get('data-stat') == 'H':
                H.append(td.get_text())
            if td.get('data-stat') == '2B':
                second_base.append(td.get_text())
            if td.get('data-stat') == '3B':
                third_base.append(td.get_text())
            if td.get('data-stat') == 'HR':
                HR.append(td.get_text())
            if td.get('data-stat') == 'RBI':
                RBI.append(td.get_text())
            if td.get('data-stat') == 'BB':
                BB.append(td.get_text())
            if td.get('data-stat') == 'IBB':
                IBB.append(td.get_text())
            if td.get('data-stat') == 'SO':
                SO.append(td.get_text())
            if td.get('data-stat') == 'HBP':
                HBP.append(td.get_text())
            if td.get('data-stat') == 'SH':
                SH.append(td.get_text())
            if td.get('data-stat') == 'SF':
                SF.append(td.get_text())
            if td.get('data-stat') == 'ROE':
                ROE.append(td.get_text())
            if td.get('data-stat') == 'GIDP':
                GIDP.append(td.get_text())
            if td.get('data-stat') == 'SB':
                SB.append(td.get_text())
            if td.get('data-stat') == 'CS':
                CS.append(td.get_text())
            if td.get('data-stat') == 'batting_avg':
                batting_avg.append(td.get_text())
            if td.get('data-stat') == 'onbase_perc':
                onbase_perc.append(td.get_text())
            if td.get('data-stat') == 'slugging_perc':
                slugging_perc.append(td.get_text())
            if td.get('data-stat') == 'onbase_plus_slugging':
                onbase_plus_slugging.append(td.get_text())
            if td.get('data-stat') == 'LOB':
                LOB.append(td.get_text())
            if td.get('data-stat') == 'batters_number':
                batters_number.append(td.get_text())
    #Pitching Data
    table = soup_pitch.find(id='all_team_pitching_gamelogs')
    tbody = table.find('tbody')
    tr_body = tbody.find_all('tr')
    IP = []
    Hits_allow = []
    Runs_allow = []
    ER = []
    UER = []
    BB_allow = []
    SO_pitch = []
    HR_allow = []
    HBP_pitch = []
    earned_run_avg = []
    batters_faced = []
    pitches = []
    strikes_total = []
    inherited_runners = []
    inherited_score = []
    SB_allow = []
    CS_pitch = []
    AB_pitch = []
    allow_2B = []
    allow_3B = []
    IBB_allow = []
    SH_allow = []
    SF_allow = []
    ROE_allow = []
    GIDP_allow = []
    pitchers_number = []
    for trb in tr_body:
        for td in trb.find_all('td'):
            if td.get('data-stat') == 'IP':
                IP.append(td.get_text())
            if td.get('data-stat') == 'H':
                Hits_allow.append(td.get_text())
            if td.get('data-stat') == 'R':
                Runs_allow.append(td.get_text())
            if td.get('data-stat') == 'ER':
                ER.append(td.get_text())
            if td.get('data-stat') == 'UER':
                UER.append(td.get_text())
            if td.get('data-stat') == 'BB':
                BB_allow.append(td.get_text())
            if td.get('data-stat') == 'SO':
                SO_pitch.append(td.get_text())
            if td.get('data-stat') == 'HR':
                HR_allow.append(td.get_text())
            if td.get('data-stat') == 'HBP':
                HBP_pitch.append(td.get_text())
            if td.get('data-stat') == 'earned_run_avg':
                earned_run_avg.append(td.get_text())
            if td.get('data-stat') == 'batters_faced':
                batters_faced.append(td.get_text())
            if td.get('data-stat') == 'pitches':
                pitches.append(td.get_text())
            if td.get('data-stat') == 'strikes_total':
                strikes_total.append(td.get_text())
            if td.get('data-stat') == 'inherited_runners':
                inherited_runners.append(td.get_text())
            if td.get('data-stat') == 'inherited_score':
                inherited_score.append(td.get_text())
            if td.get('data-stat') == 'SB':
                SB_allow.append(td.get_text())
            if td.get('data-stat') == 'CS':
                CS_pitch.append(td.get_text())
            if td.get('data-stat') == 'AB':
                AB_pitch.append(td.get_text())
            if td.get('data-stat') == '2B':
                allow_2B.append(td.get_text())
            if td.get('data-stat') == '3B':
                allow_3B.append(td.get_text())
            if td.get('data-stat') == 'IBB':
                IBB_allow.append(td.get_text())
            if td.get('data-stat') == 'SH':
                SH_allow.append(td.get_text())
            if td.get('data-stat') == 'SF':
                SF_allow.append(td.get_text())
            if td.get('data-stat') == 'ROE':
                ROE_allow.append(td.get_text())
            if td.get('data-stat') == 'GIDP':
                GIDP_allow.append(td.get_text())
            if td.get('data-stat') == 'pitchers_number':
                pitchers_number.append(td.get_text())
    #Game-by-Game
    table = soup_sched.find(id='div_team_schedule')
    tbody = table.find('tbody')
    tr_body = tbody.find_all('tr')
    rank = []
    cli = []
    for trb in tr_body:
        for td in trb.find_all('td'):
            if td.get('data-stat') == 'rank':
                rank.append(td.get_text())
            if td.get('data-stat') == 'cli':
                cli.append(td.get_text())
    # dict_val = {"game_result":game_result,
    #                   "PA":PA,
    #                   "AB":AB,
    #                   "R":R,
    #                   "H":H,
    #                   "second_base":second_base,
    #                   "third_base":third_base,
    #                   "HR":HR,
    #                   "RBI":RBI,
    #                   "BB":BB,
    #                   "IBB":IBB,
    #                   "SO":SO,
    #                   "HBP":HBP,
    #                   "SH":SH,
    #                   "SF":SF,
    #                   "ROE":ROE,
    #                   "GIDP":GIDP,
    #                   "SB":SB,
    #                   "CS":CS,
    #                   "ROE":ROE,
    #                   "batting_avg":batting_avg,
    #                   "onbase_perc":onbase_perc,
    #                   "slugging_perc":slugging_perc,
    #                   "onbase_plus_slugging":onbase_plus_slugging,
    #                   "batters_number":batters_number,
    #                   "LOB":LOB,
    #                   "IP":IP,
    #                   "Hits_allow":Hits_allow,
    #                   "Runs_allow":Runs_allow,
    #                   "ER":ER,
    #                   "UER":UER,
    #                   "BB_allow":BB_allow,
    #                   "SO_pitch":SO_pitch,
    #                   "HR_allow":HR_allow,
    #                   "HBP_pitch":HBP_pitch,
    #                   "earned_run_avg":earned_run_avg,
    #                   "batters_faced":batters_faced,
    #                   "pitches":pitches,
    #                   "strikes_total":strikes_total,
    #                   "inherited_runners":inherited_runners,
    #                   "inherited_score":inherited_score,
    #                   "SB_allow":SB_allow,
    #                   "CS_pitch":CS_pitch,
    #                   "AB_pitch":AB_pitch,
    #                   "allow_2B":allow_2B,
    #                   "allow_3B":allow_3B,
    #                   "IBB_allow":IBB_allow,
    #                   "SH_allow":SH_allow,
    #                   "SF_allow":SF_allow,
    #                   "ROE_allow":ROE_allow,
    #                   "GIDP_allow":GIDP_allow,
    #                   "pitchers_number":pitchers_number,
    #                   "game_location":team_homeORaway,
    #                   'rank':rank,
    #                   "cli":cli
    # }
    # for key, value in dict_val.items():
    #     print(key, len(value))
    return DataFrame({"game_result":game_result,
                      "PA":PA,
                      "AB":AB,
                      "R":R,
                      "H":H,
                      "second_base":second_base,
                      "third_base":third_base,
                      "HR":HR,
                      "RBI":RBI,
                      "BB":BB,
                      "IBB":IBB,
                      "SO":SO,
                      "HBP":HBP,
                      "SH":SH,
                      "SF":SF,
                      "ROE":ROE,
                      "GIDP":GIDP,
                      "SB":SB,
                      "CS":CS,
                      "ROE":ROE,
                      "batting_avg":batting_avg,
                      "onbase_perc":onbase_perc,
                      "slugging_perc":slugging_perc,
                      "onbase_plus_slugging":onbase_plus_slugging,
                      "batters_number":batters_number,
                      "LOB":LOB,
                      "IP":IP,
                      "Hits_allow":Hits_allow,
                      "Runs_allow":Runs_allow,
                      "ER":ER,
                      "UER":UER,
                      "BB_allow":BB_allow,
                      "SO_pitch":SO_pitch,
                      "HR_allow":HR_allow,
                      "HBP_pitch":HBP_pitch,
                      "earned_run_avg":earned_run_avg,
                      "batters_faced":batters_faced,
                      "pitches":pitches,
                      "strikes_total":strikes_total,
                      "inherited_runners":inherited_runners,
                      "inherited_score":inherited_score,
                      "SB_allow":SB_allow,
                      "CS_pitch":CS_pitch,
                      "AB_pitch":AB_pitch,
                      "allow_2B":allow_2B,
                      "allow_3B":allow_3B,
                      "IBB_allow":IBB_allow,
                      "SH_allow":SH_allow,
                      "SF_allow":SF_allow,
                      "ROE_allow":ROE_allow,
                      "GIDP_allow":GIDP_allow,
                      "pitchers_number":pitchers_number,
                      "game_location":team_homeORaway,
                      'rank':rank,
                      "cli":cli
    })
get_data_team('NYY',2022)