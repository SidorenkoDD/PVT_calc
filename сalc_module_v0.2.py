import pandas as pd
import numpy as np
import math as  m
import matplotlib.pyplot as plt
import time
import functools

##TODO: стоит отойти от пандас и использовать просто именные уникальные массивы. Нет смысла загонять массивы в датафреймы
# (одно и то же получаем), занимает память, дольше обработка. Потестил, разница в 5 сотых секунды, более того, дольше всего
# отрабатывает print()

##TODO: написать стандартные функции по обработке инпут-значений (oden_input -> API, p_bar -> psi)


class PVT:
    def __init__(self,rs_input_m3,oden_input, gden_input, t_input_c,
                 p1_input_bar,p2_input_bar, p_step_input_bar):
        self.rs_input_m3 = rs_input_m3
        self.oden_input = oden_input
        self.gden_input = gden_input
        self.t_input = t_input_c
        self.p1_input_bar = p1_input_bar
        self.p2_input_bar = p2_input_bar
        self.p_step_input_bar = p_step_input_bar

        '''
        === description ===
        Класс PVT принимает от пользователя базовые входные данные для расчета
        
        === input ===
        rs_input_m3 - газосодержание в м3
        oden_input - относительная плотность дегазированной нефти 
        gden_input - относительная плотность газа
        t_input_c - температура в градусах Цельсия
        p1_input_bar - начальное давление для расчета в барах
        p2_input_bar - конечное давление для расчета в барах
        p_step_input_bar - шаг давления для расчета в барах
        '''

    @functools.lru_cache()
    def RS_standing(self):
        '''
        === description ===
        Функция расчета газосодержания для переданного диапазона давлений
        === return ===
        Возвращает датаферйм давление-газосодержание
        '''
        rs_standing_res = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            # rs_standing = 1 * gden_input * m.pow((((p / 18.2) + 1.4) * m.pow(10, 0.0125 * oden - 0.00091 * t)), 1.2048)
            rs_standing = 1 * self.gden_input * m.pow(((p / 18.2) + 1.4) * m.pow(10, 0.0125 * (
                    (141.5 / self.oden_input) - 131.5) - 0.00091 * (self.t_input * 1.8 + 32)), 1.2048)
            if rs_standing < self.rs_input_m3 * 35.314 / 6.289:
                rs_standing_res.append(rs_standing)
            else:
                rs_standing_res.append(self.rs_input_m3 * 35.314 / 6.289)

        df_rs_standing_res = pd.DataFrame({'Pressure': p_massive, 'Rs_Standing': rs_standing_res})
        df_rs_standing_res['Pressure'] = df_rs_standing_res['Pressure'] / 14.5037
        df_rs_standing_res['Rs_Standing'] = df_rs_standing_res['Rs_Standing'] * 0.02831 / 0.158987
        # plt.plot(df_rs_standing_res['Pressure'], df_rs_standing_res['Rs_Standing'])
        # plt.show()
        return df_rs_standing_res

    @functools.lru_cache
    def RS_glaso(self):
        '''
        === description ===
        Функция расчета газосодержания для переданного диапазона давлений
        === return ===
        Возвращает датаферйм давление-газосодержание
        '''
        rs_glaso_res = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            # Формула Гласо для газосодержания
            a = 2.8869 - m.pow((14.1811 - 3.3093 * m.log(p, 10)), 0.5)
            rs_glaso = 1 * self.gden_input * m.pow(((((141.5 / self.oden_input) - 131.5) ** 0.989 / (
                m.pow(self.t_input * 1.8 + 32, 0.172))) * m.pow(10, a)), 1.225)
            # Логика для насыщенной и недонасыщенной ветви
            if rs_glaso < self.rs_input_m3 * 35.314 / 6.289:
                rs_glaso_res.append(rs_glaso)
            else:
                rs_glaso_res.append(self.rs_input_m3 * 35.314 / 6.289)

        df_rs_glaso_res = pd.DataFrame({'Pressure': p_massive, 'Rs_Glaso': rs_glaso_res})
        df_rs_glaso_res['Pressure'] = df_rs_glaso_res['Pressure'] / 14.5037
        df_rs_glaso_res['Rs_Glaso'] = df_rs_glaso_res['Rs_Glaso'] * 0.02831 / 0.158987

        # plt.plot(df_rs_glaso_res['Pressure'], df_rs_glaso_res['Rs_Glaso'])
        # plt.show()
        return df_rs_glaso_res

    @functools.lru_cache
    def RS_deghetto(self):
        '''
        === description ===
        Функция расчета газосодержания для переданного диапазона давлений
        === return ===
        Возвращает датаферйм давление-газосодержание
        '''
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        rs_deghetto_res = []
        for p in p_massive:
            # Корреляция де Гетто для газосодержания
            # Сверхтяжелая нефть
            if (141.5 / self.oden_input) - 131.5 <= 10:
                a = 0.01694 * ((141.5 / self.oden_input) - 131.5) - 0.00156 * (self.t_input * 1.8 + 32)
                rs_deghetto = 1 * self.gden_input * m.pow(((p / 10.7025) * m.pow(10, a)), 1.1128)
                # Логика для насыщенной и недонасыщенной ветви
                if rs_deghetto < self.rs_input_m3 * 35.314 / 6.289:
                    rs_deghetto_res.append(rs_deghetto)
                else:
                    rs_deghetto_res.append(self.rs_input_m3 * 35.314 / 6.289)
            # Тяжелая нефть
            if 10 < ((141.5 / self.oden_input) - 131.5) <= 22.3:
                a = 0.0142 * (141.5 / self.oden_input) - 131. - 0.0020 * (self.t_input * 1.8 + 32)
                rs_deghetto = 1 * self.gden_input * m.pow(((p / 15.7286) * m.pow(10, a)), 1 / 0.7885)
                if rs_deghetto < self.rs_input_m3 * 35.314 / 6.289:
                    rs_deghetto_res.append(rs_deghetto)
                else:
                    rs_deghetto_res.append(self.rs_input_m3 * 35.314 / 6.289)
            # Средняя нефть
            if 22.3 < ((141.5 / self.oden_input) - 131.5) <= 31.1:
                a = (7.4576 * ((141.5 / self.oden_input) - 131.5)) / ((self.t_input * 1.8 + 32) + 460)
                rs_deghetto = 0.1 * m.pow(self.gden_input, 0.2556) * m.pow(p, 0.9868) * m.pow(10, a)
                if rs_deghetto < self.rs_input_m3 * 35.314 / 6.289:
                    rs_deghetto_res.append(rs_deghetto)
                else:
                    rs_deghetto_res.append(self.rs_input_m3 * 35.314 / 6.289)
            # Легкая нефть
            if 31.1 < ((141.5 / self.oden_input) - 131.5):
                a = 0.0148 * ((141.5 / self.oden_input) - 131.5) - 0.0009 * (self.t_input * 1.8 + 32)
                rs_deghetto = 1 * self.gden_input * m.pow((p * m.pow(10, a) / 31.7648), 1 / 0.7857)
                if rs_deghetto < self.rs_input_m3 * 35.314 / 6.289:
                    rs_deghetto_res.append(rs_deghetto)
                else:
                    rs_deghetto_res.append(self.rs_input_m3 * 35.314 / 6.289)
        df_rs_deghetto_res = pd.DataFrame({'Pressure': p_massive, 'Rs_De_Ghetto': rs_deghetto_res})
        df_rs_deghetto_res['Pressure'] = df_rs_deghetto_res['Pressure'] / 14.5037
        df_rs_deghetto_res['Rs_De_Ghetto'] = df_rs_deghetto_res['Rs_De_Ghetto'] * 0.02831 / 0.158987

        # plt.plot(df_rs_deghetto_res['Pressure'], df_rs_deghetto_res['Rs_De_Ghetto'])
        # plt.show()

        return df_rs_deghetto_res

    # Источник: https://www.ihsenergy.ca/support/documentation_ca/Harmony/content/html_files/reference_material/calculations_and_correlations/oil_correlations.htm
    @functools.lru_cache
    def RS_vb(self):
        '''
        === description ===
        Функция расчета газосодержания для переданного диапазона давлений
        === return ===
        Возвращает датаферйм давление-газосодержание
        '''
        rs_vb_res = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            if ((141.5 / self.oden_input) - 131.5) <= 30:
                c1 = 0.0362
                c2 = 1.0937
                c3 = 25.7240
            else:
                c1 = 0.0178
                c2 = 1.1870
                c3 = 23.9310
            # Версия тНавигатора не корректна! (крутой взлет до насыщения за 20-30 бар)
            # a = (c3 * oden)/(t+460)
            # rs_vb = (1/c1) + gden_input * pow((p-14.7), c2) * pow(10, a)
            rs_vb = c1 * self.gden_input * m.pow(p, c2) * m.pow(m.e, c3 * (
                    ((141.5 / self.oden_input) - 131.5) / ((self.t_input * 1.8 + 32) + 460)))
            if rs_vb < self.rs_input_m3 * 35.314 / 6.289:
                rs_vb_res.append(rs_vb)
            else:
                rs_vb_res.append(self.rs_input_m3 * 35.314 / 6.289)
        df_rs_vb_res = pd.DataFrame({'Pressure': p_massive, 'Rs_Vasquez&Beggs': rs_vb_res})
        df_rs_vb_res['Rs_Vasquez&Beggs'] = df_rs_vb_res['Rs_Vasquez&Beggs'] * 0.02831 / 0.158987
        df_rs_vb_res['Pressure'] = df_rs_vb_res['Pressure'] / 14.5037

        # plt.plot(df_rs_vb_res['Pressure'], df_rs_vb_res['Rs_VasquezBeggs'])
        # plt.show()

        return df_rs_vb_res

    @functools.lru_cache
    def RS_pf(self):
        '''
        === description ===
        Функция расчета газосодержания для переданного диапазона давлений
        === return ===
        Возвращает датаферйм давление-газосодержание
        '''
        rs_pf_res = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037, self.p_step_input_bar * 14.5037)
        for p in p_massive:
            a = 7.916 * m.pow(10, -4) * m.pow(((141.5 / self.oden_input) - 131.5), 1.541) - 4.561 * m.pow(10,
                                                                                                          -5) * m.pow(
                (self.t_input * 1.8 + 32), 1.3911)
            rs_pf = 1 * m.pow(((p / 112.727) + 12.34) * m.pow(self.gden_input, 0.8439) * m.pow(10, a), 1.73184)
            if rs_pf < self.rs_input_m3 * 35.314 / 6.289:
                rs_pf_res.append(rs_pf)
            else:
                rs_pf_res.append(self.rs_input_m3 * 35.314 / 6.289)

        df_rs_pf_res = pd.DataFrame({'Pressure': p_massive, 'Rs_Petorsky&Farshad': rs_pf_res})
        df_rs_pf_res['Rs_Petorsky&Farshad'] = df_rs_pf_res['Rs_Petorsky&Farshad'] * 0.02831 / 0.158987
        df_rs_pf_res['Pressure'] = df_rs_pf_res['Pressure'] / 14.5037

        # plt.plot(df_rs_vb_res['Pressure'], df_rs_vb_res['Rs_Petorsky&Farshad'])
        # plt.show()

        return df_rs_pf_res

    @functools.lru_cache
    def RS_lasater(self):
        '''
        === description ===
        Функция расчета газосодержания для переданного диапазона давлений
        === return ===
        Возвращает датаферйм давление-газосодержание
        '''
        rs_lasater_res = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037, self.p_step_input_bar * 14.5037)
        for p in p_massive:
            mwo = (677.3893 - 13.2161 * ((141.5 / self.oden_input) - 131.5) + 0.024775 * m.pow(
                ((141.5 / self.oden_input) - 131.5), 2)
                   + 0.00067851 * m.pow(((141.5 / self.oden_input) - 131.5), 3))
            yg = 0.08729793 + 0.37912718 * m.log(((p * self.gden_input) / ((self.t_input * 1.8 + 32) + 460)) + 0.769066)
            rs = 132755 * (yg / (1 - yg)) * (self.oden_input / mwo)
            if rs < self.rs_input_m3 * 35.314 / 6.289:
                rs_lasater_res.append(rs)
            else:
                rs_lasater_res.append(self.rs_input_m3 * 35.314 / 6.289)

        df_rs_lasater_res = pd.DataFrame({'Pressure': p_massive, 'Rs_Lasater': rs_lasater_res})
        df_rs_lasater_res['Rs_Lasater'] = df_rs_lasater_res['Rs_Lasater'] * 0.02831 / 0.158987
        df_rs_lasater_res['Pressure'] = df_rs_lasater_res['Pressure'] / 14.5037

        # plt.plot(df_rs_lasater_res['Pressure'], df_rs_lasater_res['Rs_Lasater'])
        # plt.show()
        return df_rs_lasater_res

    @functools.lru_cache
    def RS_standing_cal(self,p):
        '''
        === description ===
        Вспомогательная функция расчета газосодержания для точечного расчета газосодержания.
        Используется в других функциях: объемный коэффициент, вязкость
        === input ===
        p - давление для расчета
        === return ===
        Возвращает значение газосодержания
        '''
        rs_standing = 1 * self.gden_input * m.pow(((p / 18.2) + 1.4) * m.pow(10, 0.0125 * (
                        (141.5 / self.oden_input) - 131.5) - 0.00091 * (self.t_input * 1.8 + 32)), 1.2048)
        if rs_standing < self.rs_input_m3 * 35.314 / 6.289:
            return rs_standing
        else:
            return (self.rs_input_m3 * 35.314 / 6.289)

    @functools.lru_cache
    def RS_glaso_cal(self,p):
        '''
        === description ===
        Вспомогательная функция расчета газосодержания для точечного расчета газосодержания.
        Используется в других функциях: объемный коэффициент, вязкость
        === input ===
        p - давление для расчета
        === return ===
        Возвращает значение газосодержания
        '''
        a = 2.8869 - m.pow((14.1811 - 3.3093 * m.log(p, 10)), 0.5)
        rs_glaso = 1 * self.gden_input * m.pow(((((141.5 / self.oden_input) - 131.5) ** 0.989 / (m.pow(self.t_input * 1.8 + 32, 0.172))) * m.pow(10, a)), 1.225)
        # Логика для насыщенной и недонасыщенной ветви
        if rs_glaso < self.rs_input_m3 * 35.314 / 6.289:
            return (rs_glaso)
        else:
            return (self.rs_input_m3 * 35.314 / 6.289)

    @functools.lru_cache
    def RS_deghetto_cal(self,p):
        '''
        === description ===
        Вспомогательная функция расчета газосодержания для точечного расчета газосодержания.
        Используется в других функциях: объемный коэффициент, вязкость
        === input ===
        p - давление для расчета
        === return ===
        Возвращает значение газосодержания
        '''
        # Сверхтяжелая нефть
        if (141.5 / self.oden_input) - 131.5 <= 10:
            a = 0.01694 * ((141.5 / self.oden_input) - 131.5) - 0.00156 * (self.t_input * 1.8 + 32)
            rs_deghetto = 1 * self.gden_input * m.pow(((p / 10.7025) * m.pow(10, a)), 1.1128)
            if rs_deghetto < self.rs_input_m3 * 35.314 / 6.289:
                return (rs_deghetto)
            else:
                return (self.rs_input_m3 * 35.314 / 6.289)
        # Тяжелая нефть
        if 10 < ((141.5 / self.oden_input) - 131.5) <= 22.3:
            a = 0.0142 * (141.5 / self.oden_input) - 131. - 0.0020 * (self.t_input * 1.8 + 32)
            rs_deghetto = 1 * self.gden_input * m.pow(((p / 15.7286) * m.pow(10, a)), 1 / 0.7885)
            if rs_deghetto < self.rs_input_m3 * 35.314 / 6.289:
                return (rs_deghetto)
            else:
                return (self.rs_input_m3 * 35.314 / 6.289)

        # Средняя нефть
        if 22.3 < ((141.5 / self.oden_input) - 131.5) <= 31.1:
            a = (7.4576 * ((141.5 / self.oden_input) - 131.5)) / ((self.t_input * 1.8 + 32) + 460)
            rs_deghetto = 0.1 * m.pow(self.gden_input, 0.2556) * m.pow(p, 0.9868) * m.pow(10, a)
            if rs_deghetto < self.rs_input_m3 * 35.314 / 6.289:
                return (rs_deghetto)
            else:
                return (self.rs_input_m3 * 35.314 / 6.289)

        # Легкая нефть
        if 31.1 < ((141.5 / self.oden_input) - 131.5):
            a = 0.0148 * ((141.5 / self.oden_input) - 131.5) - 0.0009 * (self.t_input * 1.8 + 32)
            rs_deghetto = 1 * self.gden_input * m.pow((p * m.pow(10, a) / 31.7648), 1 / 0.7857)
            if rs_deghetto < self.rs_input_m3 * 35.314 / 6.289:
                return (rs_deghetto)
            else:
                return (self.rs_input_m3 * 35.314 / 6.289)




    # Источник: https://www.ihsenergy.ca/support/documentation_ca/Harmony/content/html_files/reference_material/calculations_and_correlations/oil_correlations.htm
    @functools.lru_cache
    def RS_vb_cal(self,p):
        '''
        === description ===
        Вспомогательная функция расчета газосодержания для точечного расчета газосодержания.
        Используется в других функциях: объемный коэффициент, вязкость
        === input ===
        p - давление для расчета
        === return ===
        Возвращает значение газосодержания
        '''
        if ((141.5 / self.oden_input) - 131.5) <= 30:
            c1 = 0.0362
            c2 = 1.0937
            c3 = 25.7240
        else:
            c1 = 0.0178
            c2 = 1.1870
            c3 = 23.9310
        # Версия тНавигатора не корректна! (крутой взлет до насыщения за 20-30 бар)
        # a = (c3 * oden)/(t+460)
        # rs_vb = (1/c1) + gden_input * pow((p-14.7), c2) * pow(10, a)
        rs_vb = c1 * self.gden_input * m.pow(p, c2) * m.pow(m.e, c3 * (
                    ((141.5 / self.oden_input) - 131.5) / ((self.t_input * 1.8 + 32) + 460)))
        if rs_vb < self.rs_input_m3 * 35.314 / 6.289:
            return(rs_vb)
        else:
            return (self.rs_input_m3 * 35.314 / 6.289)

    @functools.lru_cache
    def RS_pf_cal(self,p):
        '''
        === description ===
        Вспомогательная функция расчета газосодержания для точечного расчета газосодержания.
        Используется в других функциях: объемный коэффициент, вязкость
        === input ===
        p - давление для расчета
        === return ===
        Возвращает значение газосодержания
        '''
        a = 7.916 * m.pow(10,-4) * m.pow(((141.5/self.oden_input)-131.5),1.541) - 4.561 * m.pow(10,-5) * m.pow((self.t_input*1.8+32),1.3911)
        rs_pf = 1 * m.pow(((p/112.727)+12.34)*m.pow(self.gden_input,0.8439)*m.pow(10,a), 1.73184)
        if rs_pf < self.rs_input_m3*35.314/6.289:
            return (rs_pf)
        else:
            return (self.rs_input_m3*35.314/6.289)

    @functools.lru_cache
    def RS_lasater_cal(self,p):
        '''
        === description ===
        Вспомогательная функция расчета газосодержания для точечного расчета газосодержания.
        Используется в других функциях: объемный коэффициент, вязкость
        === input ===
        p - давление для расчета
        === return ===
        Возвращает значение газосодержания
        '''
        mwo = (677.3893- 13.2161 * ((141.5/self.oden_input)-131.5) + 0.024775 * m.pow(((141.5/self.oden_input)-131.5),2)
              + 0.00067851 *m.pow(((141.5/self.oden_input)-131.5),3))
        yg = 0.08729793+0.37912718*m.log(((p*self.gden_input)/((self.t_input*1.8+32)+460))+0.769066)
        rs = 132755 *(yg/(1-yg))*(self.oden_input/mwo)
        if rs < self.rs_input_m3 * 35.314 / 6.289:
            return (rs)
        else:
            return self.rs_input_m3 * 35.314 / 6.289

    '''
        ================== Давление насыщения ==================
    '''

    @functools.lru_cache
    def Pb_standing(self):
        '''
        === description ===
        Вспомогательная функция расчета давления насыщения.
        Используется в других функциях: объемный коэффициент, вязкость.
        Внутри вызывает расчет газосодержания и итерируется по датафрейму газосодержание-давление
        === return ===
        Возвращает значение давления насыщения
        '''
        self.RS_standing()
        for rs in self.RS_standing()['Rs_Standing']:
            if rs - self.rs_input_m3 < 0.02:
                pb_standing = self.RS_standing().loc[self.RS_standing()['Rs_Standing'] == rs, 'Pressure'].iloc[0]
            else:
                continue
        return pb_standing

    @functools.lru_cache
    def Pb_glaso(self):
        '''
        === description ===
        Вспомогательная функция расчета давления насыщения.
        Используется в других функциях: объемный коэффициент, вязкость.
        Внутри вызывает расчет газосодержания и итерируется по датафрейму газосодержание-давление
        === return ===
        Возвращает значение давления насыщения
        '''
        for rs in self.RS_glaso()['Rs_Glaso']:
            if rs - self.rs_input_m3 < 0.02:
                pb_glaso = self.RS_glaso().loc[self.RS_glaso()['Rs_Glaso'] == rs, 'Pressure'].iloc[0]
            else:
                continue
        return pb_glaso

    @functools.lru_cache
    def Pb_deghetto(self):
        '''
        === description ===
        Вспомогательная функция расчета давления насыщения.
        Используется в других функциях: объемный коэффициент, вязкость.
        Внутри вызывает расчет газосодержания и итерируется по датафрейму газосодержание-давление
        === return ===
        Возвращает значение давления насыщения
        '''
        for rs in self.RS_deghetto()['Rs_De_Ghetto']:
            if rs - self.rs_input_m3 < 0.02:
                pb_deghetto = self.RS_deghetto().loc[self.RS_deghetto()['Rs_De_Ghetto'] == rs, 'Pressure'].iloc[0]
            else:
                continue
        return pb_deghetto

    @functools.lru_cache
    def Pb_vb(self):
        '''
        === description ===
        Вспомогательная функция расчета давления насыщения.
        Используется в других функциях: объемный коэффициент, вязкость.
        Внутри вызывает расчет газосодержания и итерируется по датафрейму газосодержание-давление
        === return ===
        Возвращает значение давления насыщения
        '''
        for rs in self.RS_vb()['Rs_Vasquez&Beggs']:
            if rs - self.rs_input_m3 < 0.02:
                pb_vb = self.RS_vb().loc[self.RS_vb()['Rs_Vasquez&Beggs'] == rs, 'Pressure'].iloc[0]
            else:
                continue
        return pb_vb

    @functools.lru_cache
    def Pb_pf(self):
        '''
        === description ===
        Вспомогательная функция расчета давления насыщения.
        Используется в других функциях: объемный коэффициент, вязкость.
        Внутри вызывает расчет газосодержания и итерируется по датафрейму газосодержание-давление
        === return ===
        Возвращает значение давления насыщения
        '''
        for rs in self.RS_pf()['Rs_Petorsky&Farshad']:
            if rs - self.rs_input_m3 < 0.02:
                pb_pf = self.RS_pf().loc[self.RS_pf()['Rs_Petorsky&Farshad'] == rs, 'Pressure'].iloc[0]
            else:
                continue
        return pb_pf

    @functools.lru_cache
    def Pb_lasater(self):
        '''
        === description ===
        Вспомогательная функция расчета давления насыщения.
        Используется в других функциях: объемный коэффициент, вязкость.
        Внутри вызывает расчет газосодержания и итерируется по датафрейму газосодержание-давление
        === return ===
        Возвращает значение давления насыщения
        '''
        for rs in self.RS_lasater()['Rs_Lasater']:
            if rs - self.rs_input_m3 < 0.02:
                pb_lasater = self.RS_lasater().loc[self.RS_lasater()['Rs_Lasater'] == rs, 'Pressure'].iloc[0]
            else:
                continue
        return pb_lasater

    '''
    ================== Объемный коэффициент нефти (насыщенная ветвь) ==================
    '''

    @functools.lru_cache
    def BOsat_standing(self, rs_corr):
        '''
        === description ===
        Функция для расчета объемного коэффициента насыщенной ветви
        === input ===
        rs_corr - выбранная корреляция для газосодержания
        === return ===
        Возвращает датафрейм: объемный коэффициент для насыщенной ветви - давление
        '''
        rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                        'Glaso': self.RS_glaso_cal,
                        'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                        'Lasater': self.RS_lasater_cal}

        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        bosat_standing_res = []
        p_sat = []
        for p in p_massive:
            rs = rs_dict_help[rs_corr](p)
            if rs < (self.rs_input_m3 * 35.314 / 6.289) - 0.01:
                p_sat.append(p)
                bo_sat = 0.9759 + 0.000120 * m.pow((rs * (
                    m.pow(self.gden_input / ((141.5 / self.oden_input) - 131.5), 0.5)) + 1.25 * (
                                                                self.t_input * 1.8 + 32)), 1.2)
                bosat_standing_res.append(bo_sat)
        df_bosat_standing_res = pd.DataFrame({'Pressure': p_sat, 'bo': bosat_standing_res})
        df_bosat_standing_res['Pressure'] = df_bosat_standing_res['Pressure'] / 14.5037


        return df_bosat_standing_res, df_bosat_standing_res['bo'].iloc[-1]

    def BOsat_vb(self, rs_corr):
        '''
               === description ===
               Функция для расчета объемного коэффициента насыщенной ветви
               === input ===
               rs_corr - выбранная корреляция для газосодержания
               === return ===
               Возвращает датафрейм: объемный коэффициент для насыщенной ветви - давление
               '''
        rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                        'Glaso': self.RS_glaso_cal,
                        'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                        'Lasater': self.RS_lasater_cal}
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        bosat_vb_res = []
        p_sat = []
        if ((141.5 / self.oden_input) - 131.5) <= 30:
            a1 = 4.677E-04
            a2 = 1.751E-05
            a3 = -1.811E-08
        else:
            a1 = 4.670E-04
            a2 = 1.100E-05
            a3 = 1.337E-09
        for p in p_massive:
            rs = rs_dict_help[rs_corr](p)
            if rs < (self.rs_input_m3 * 35.314 / 6.289) - 0.01:
                p_sat.append(p)
                bo_sat = 1 + a1 * rs + a2 * (self.t_input * 1.8 + 32 - 60) * (self.oden_input/self.gden_input) + a3 * rs * (self.t_input * 1.8 + 32 - 60) * (self.oden_input/self.gden_input)
                bosat_vb_res.append(bo_sat)
        df_bosat_standing_res = pd.DataFrame({'Pressure': p_sat, 'bo': bosat_vb_res})
        df_bosat_standing_res['Pressure'] = df_bosat_standing_res['Pressure'] / 14.5037

        return df_bosat_standing_res, df_bosat_standing_res['bo'].iloc[-1]



    '''
        ================== Объемный коэффициент нефти (недонасыщенная ветвь) ==================
    '''

    def BOunsat_standing(self, rs_corr, bosat_corr):
        '''
        === description ===
        Функция для расчета объемного коэффициента недонасыщенной ветви
        === input ===
        rs_corr - выбранная корреляция для газосодержания
        === return ===
        Возвращает датафрейм объемный коэффициент (насыщенная + недонасыщенная ветвь) - давление
        '''
        pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                        'Glaso': self.Pb_glaso(),
                        'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                        'Lasater': self.Pb_lasater()}
        bosat_dict_help = {'Standing': self.BOsat_standing(rs_corr),
                           'Vasquez_Beggs':self.BOsat_vb(rs_corr)}
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        bounsat_standing_res = []
        p_unsat = []
        pb_searched = pb_dict_help[rs_corr] * 14.5037
        bob = bosat_dict_help[bosat_corr][1]
        for p in p_massive:
            if p > pb_searched:
                p_unsat.append(p)
                a = (4.1646 * m.pow(10, -7) * m.pow((self.rs_input_m3 * 35.314 / 6.289), 0.69357) * m.pow(
            self.gden_input, 0.1885) * m.pow(((141.5 / self.oden_input) - 131.5), 0.3272) * m.pow(
            (self.t_input * 1.8) + 32, 0.6729))
                bo_unsat = bob * m.exp(-a * (m.pow(p, 0.4094) - m.pow(pb_searched, 0.4094)))
                bounsat_standing_res.append(bo_unsat)

        df_bounsat_standing_res = pd.DataFrame({'Pressure': p_unsat, 'bo': bounsat_standing_res})
        df_bounsat_standing_res['Pressure'] = df_bounsat_standing_res['Pressure'] / 14.5037

        bo_sat = bosat_dict_help[bosat_corr][0]
        bo_res = pd.concat([bo_sat, df_bounsat_standing_res], axis=0)
        #plt.plot(bo_res['Pressure'], bo_res['bo'])
        #plt.show()

        return df_bounsat_standing_res

    def Bounsat_vb(self, rs_corr,bosat_corr):
        '''
        === description ===
        Функция для расчета объемного коэффициента недонасыщенной ветви
        === input ===
        rs_corr - выбранная корреляция для газосодержания
        === return ===
        Возвращает датафрейм: объемный коэффициент для недонасыщенной ветви - давление
        '''

        pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                        'Glaso': self.Pb_glaso(),
                        'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                        'Lasater': self.Pb_lasater()}
        bosat_dict_help = {'Standing': self.BOsat_standing(rs_corr),
                           'Vasquez_Beggs': self.BOsat_vb(rs_corr)}
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        bounsat_vb_res = []
        p_unsat = []
        pb = pb_dict_help[rs_corr] * 14.5037
        bob = bosat_dict_help[bosat_corr][1]
        for p in p_massive:
            if p > pb:
                p_unsat.append(p)
                c = (-1433 + 5 * (self.rs_input_m3* 35.314 / 6.289) +17.2 *(self.t_input * 1.8 + 32) - 1180 * self.gden_input + 12.61 * self.oden_input)/(p * m.pow(10,5))
                bo_unsat = bob * m.exp(c *(pb - p))
                bounsat_vb_res.append(bo_unsat)

        df_bounsat_vb_res = pd.DataFrame({'Pressure': p_unsat, 'bo': bounsat_vb_res})
        df_bounsat_vb_res['Pressure'] = df_bounsat_vb_res['Pressure'] / 14.5037
        bo_sat = bosat_dict_help[bosat_corr][0]
        bo_res = pd.concat([bo_sat, df_bounsat_vb_res], axis=0)
        plt.plot(bo_res['Pressure'], bo_res['bo'])
        plt.show()

        return df_bounsat_vb_res



        ##TODO: работает некорректно, исправить
    def BO_vb(self, rs_corr):
        rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                        'Glaso': self.RS_glaso_cal,
                        'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                        'Lasater': self.RS_lasater_cal}
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        bo_vb_res = []
        for p in p_massive:
            rs = rs_dict_help[rs_corr](p)
            if rs < (self.rs_input_m3 * 35.314 / 6.289) - 0.02:
                if ((141.5 / self.oden_input) - 131.5) <= 30:
                    c1 = 4.677 * m.pow(10, -4)
                    c2 = 1.751 * m.pow(10, -5)
                    c3 = (-1.81 * m.pow(10, -8))
                else:
                    c1 = 4.67 * m.pow(10, -4)
                    c2 = 1.1 * m.pow(10, -5)
                    c3 = (1.337 * m.pow(10, -9))

                bob = 1 + c1 * rs + (((self.t_input * 1.8) + 32) - 60) * (
                            ((141.5 / self.oden_input) - 131.5) / self.gden_input) + (c2 + c3 * rs)
                bo_vb_res.append(bob)
            else:
                co = m.pow(10, -5) * (((5 * self.rs_input_m3 * 35.314 / 6.289) * 17.2 * (
                            (self.t_input * 1.8) + 32) - 1180 * self.gden_input + 12.61 * (
                                                   (141.5 / self.oden_input) - 131.5) - 1433) / p)
                bo_vb_res.append(co)

        df_bo_vb_res = pd.DataFrame({'Pressure': p_massive, 'Bo_VB': bo_vb_res})
        df_bo_vb_res['Pressure'] = df_bo_vb_res['Pressure'] / 14.5037

        return df_bo_vb_res


    '''
    ================== Вязкость дегазированной нефти ==================
    '''

    @functools.lru_cache
    def MUod_standing(self):
        a_muod = m.pow(10, 1.8653 - 0.025086 * ((141.5 / self.oden_input) - 131.5) - 0.5644 * m.log(
            ((self.t_input * 1.8) + 32), 10))
        muod = m.pow(10, a_muod) - 1
        return muod

    @functools.lru_cache
    def MUod_br(self):
        z = 3.0324 - 0.02023 * ((141.5 / self.oden_input) - 131.5)
        y = m.pow(10, z)
        x = y * m.pow(((self.t_input * 1.8) + 32), -1.163)
        muod = m.pow(10, x) - 1
        return muod

    @functools.lru_cache
    def MUod_krt_shdt(self):
        c = 16 * m.pow(10, 9) * m.pow(((self.t_input * 1.8) + 32), -2.8177)
        d = 5.7526 * m.log(((self.t_input * 1.8) + 32), 10) - 26.9718
        muod = c * m.pow(m.log(((141.5 / self.oden_input) - 131.5), 10), d)
        return muod

    @functools.lru_cache
    def MUod_pf(self):
        c = 2.3511 * m.pow(10,7) * m.pow(((self.t_input * 1.8) + 32),-2.10255)
        d = 4.59388 * m.log10(((self.t_input * 1.8) + 32)) - 22.82792
        muod = c * m.pow(m.log10(((141.5 / self.oden_input) - 131.5)),d)
        return muod

    @functools.lru_cache
    def MUod_De_Ghetto(self):
        if ((141.5 / self.oden_input) - 131.5) <= 10:
            y = 1.90296 - 0.012619 * ((141.5 / self.oden_input) - 131.5) - 0.61748 * m.log10(((self.t_input * 1.8) + 32))
            x = m.pow(10,y)
            muod = m.pow(10,x) - 1
        if 10 < ((141.5 / self.oden_input) - 131.5) <= 22.3:
            y = 2.06492 - 0.0179 * ((141.5 / self.oden_input) - 131.5) - 0.70226 * m.log10(((self.t_input * 1.8) + 32))
            x = m.pow(10,y)
            muod = m.pow(10,x) - 1
        if 22.3 < ((141.5 / self.oden_input) - 131.5) <= 31.1:
            c = 220.15 * m.pow(10,9) * m.pow(((self.t_input * 1.8)+32),-3.556)
            d = 12.5428 * m.log10(((self.t_input * 1.8)+32)) - 45.7874
            muod = c * (m.pow(m.log10(((141.5 / self.oden_input) - 131.5)),d))
        if 31.1 < ((141.5 / self.oden_input) - 131.5):
            y = 1.67083 - 0.017628 * ((141.5 / self.oden_input) - 131.5) - 0.61304 * m.log10(((self.t_input * 1.8) + 32))
            x = m.pow(10, y)
            muod = m.pow(10, x) - 1

        return muod


    @functools.lru_cache
    def MUod_Glaso(self):
        c = 3.141 * m.pow(10, 10) * m.pow(((self.t_input * 1.8) + 32), -3.444)
        d = 10.313 * m.log10(((self.t_input * 1.8) + 32)) - 36.447
        muod = c * (m.pow(m.log10(((141.5 / self.oden_input) - 131.5)),d))
        return muod

    @functools.lru_cache
    def MUod_Elsharkawy_Alikhan(self):
        y = 2.16924 - 0.02525 * ((141.5 / self.oden_input) - 131.5) - 0.68875 * m.log10(((self.t_input * 1.8) + 32))
        x = m.pow(10, y)
        muod = m.pow(10, x) - 1
        return muod

    def MUod_Hossian(self):
        pass

    '''
    ================== Вязкость нефти (насыщенная ветвь) ==================
    '''


    @functools.lru_cache
    def MUsat_Standing(self, RS_corr,muod_corr):
        musat_standing_res = []
        p_sat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                            'Glaso': self.RS_glaso_cal,
                            'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                            'Lasater': self.RS_lasater_cal}
            pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                            'Glaso': self.Pb_glaso(),
                            'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                            'Lasater': self.Pb_lasater()}
            muod_dict_help = {'Standing': self.MUod_standing(),'De_Ghetto': self.MUod_De_Ghetto(),
                              'Glaso': self.MUod_Glaso(), 'Elsharkawy_Alikhan': self.MUod_Elsharkawy_Alikhan(),
                              'Petorsky_Farshad': self.MUod_pf(), 'Beggs_Robinson': self.MUod_br(),
                              'Kartomodjo_Shmidt':self.MUod_krt_shdt()}
            if p <= pb_dict_help[RS_corr] * 14.5037:
                p_sat.append(p)
                rs = rs_dict_help[RS_corr](p)
                c = 8.62 * m.pow(10, -5) * rs
                d = 1.1 * m.pow(10, -3) * rs
                e = 3.74 * m.pow(10, -3) * rs
                b = (0.68 / m.pow(10, c)) + (0.25 / (m.pow(10, d))) + (0.062 / (m.pow(10, e)))
                a = rs * (2.2 * m.pow(10, -7) * rs - 7.4 * m.pow(10, -4))
                muob = m.pow(10, a) * muod_dict_help[muod_corr] ** b
                musat_standing_res.append(muob)
        df_mu_standing_res_sat = pd.DataFrame({'Pressure': p_sat, 'mu': musat_standing_res})
        df_mu_standing_res_sat['Pressure'] = df_mu_standing_res_sat['Pressure'] / 14.5037
        #plt.plot(df_mu_standing_res_sat['Pressure'], musat_standing_res)
        #plt.show()
        return df_mu_standing_res_sat , df_mu_standing_res_sat['mu'].iloc[-1]


    @functools.lru_cache
    def MUsat_krt_shdt(self, RS_corr, muod_corr):
        musat_krt_shdt_res = []
        p_sat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                            'Glaso': self.RS_glaso_cal,
                            'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                            'Lasater': self.RS_lasater_cal}
            pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                            'Glaso': self.Pb_glaso(),
                            'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                            'Lasater': self.Pb_lasater()}
            muod_dict_help = {'Standing': self.MUod_standing(), 'De_Ghetto': self.MUod_De_Ghetto(),
                              'Glaso': self.MUod_Glaso(), 'Elsharkawy_Alikhan': self.MUod_Elsharkawy_Alikhan(),
                              'Petorsky_Farshad': self.MUod_pf(), 'Beggs_Robinson': self.MUod_br(),
                              'Kartomodjo_Shmidt': self.MUod_krt_shdt()}
            if p <= pb_dict_help[RS_corr] * 14.5037:
                p_sat.append(p)
                rs = rs_dict_help[RS_corr](p)
                a = 0.2001 + 0.8428 * m.pow(10, -8.45 * m.pow(10, -4) * rs)
                f = a * m.pow(muod_dict_help[muod_corr], 0.43 + 0.5165 * m.pow(10, - 8.1 * 10 ** -4 * rs))
                muob = -0.06821 + 0.9824 * f + 4.034 * m.pow(10, -4) * m.pow(f, 2)
                musat_krt_shdt_res.append(muob)
        df_mu_krt_shdt_res_sat = pd.DataFrame({'Pressure': p_sat, 'mu': musat_krt_shdt_res})
        df_mu_krt_shdt_res_sat['Pressure'] = df_mu_krt_shdt_res_sat['Pressure'] / 14.5037
        #plt.plot(df_mu_krt_shdt_res_sat['Pressure'], musat_krt_shdt_res)
        #plt.show()

        return df_mu_krt_shdt_res_sat, df_mu_krt_shdt_res_sat['mu'].iloc[-1]

    @functools.lru_cache
    def Musat_beggs_robinson(self, RS_corr, muod_corr):
        musat_br_res = []
        p_sat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                            'Glaso': self.RS_glaso_cal,
                            'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                            'Lasater': self.RS_lasater_cal}
            pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                            'Glaso': self.Pb_glaso(),
                            'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                            'Lasater': self.Pb_lasater()}
            muod_dict_help = {'Standing': self.MUod_standing(), 'De_Ghetto': self.MUod_De_Ghetto(),
                              'Glaso': self.MUod_Glaso(), 'Elsharkawy_Alikhan': self.MUod_Elsharkawy_Alikhan(),
                              'Petorsky_Farshad': self.MUod_pf(), 'Beggs_Robinson': self.MUod_br(),
                              'Kartomodjo_Shmidt': self.MUod_krt_shdt()}
            if p <= pb_dict_help[RS_corr] * 14.5037:
                p_sat.append(p)
                rs = rs_dict_help[RS_corr](p)
                a = 10.715 * m.pow((100 + rs),-0.515)
                b = 5.44 * m.pow((100 + rs),-0.338)
                muob = a * m.pow(muod_dict_help[muod_corr],b)
                musat_br_res.append(muob)

        df_mu_br_res_sat = pd.DataFrame({'Pressure': p_sat, 'mu': musat_br_res})
        df_mu_br_res_sat['Pressure'] = df_mu_br_res_sat['Pressure'] / 14.5037

        return df_mu_br_res_sat, df_mu_br_res_sat['mu'].iloc[-1]

    @functools.lru_cache
    def Musat_Chew_Connanly(self, rs_corr, muod_corr):
        musat_chc_res = []
        p_sat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                            'Glaso': self.RS_glaso_cal,
                            'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                            'Lasater': self.RS_lasater_cal}
            pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                            'Glaso': self.Pb_glaso(),
                            'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                            'Lasater': self.Pb_lasater()}
            muod_dict_help = {'Standing': self.MUod_standing(), 'De_Ghetto': self.MUod_De_Ghetto(),
                              'Glaso': self.MUod_Glaso(), 'Elsharkawy_Alikhan': self.MUod_Elsharkawy_Alikhan(),
                              'Petorsky_Farshad': self.MUod_pf(), 'Beggs_Robinson': self.MUod_br(),
                              'Kartomodjo_Shmidt': self.MUod_krt_shdt()}
            if p <= pb_dict_help[rs_corr] * 14.5037:
                p_sat.append(p)
                rs = rs_dict_help[rs_corr](p)
                a = 0.2 + (0.8/m.pow(10,0.000852 * rs))
                b = 0.482 + (0.518/m.pow(10,0.000777 * rs))
                muob = a * m.pow(muod_dict_help[muod_corr],b)
                musat_chc_res.append(muob)
        df_mu_chc_res_sat = pd.DataFrame({'Pressure': p_sat, 'mu': musat_chc_res})
        df_mu_chc_res_sat['Pressure'] = df_mu_chc_res_sat['Pressure'] / 14.5037

        return df_mu_chc_res_sat, df_mu_chc_res_sat['mu'].iloc[-1]

    @functools.lru_cache
    def Musat_Elsharkawy_Alikhan(self,rs_corr,muod_corr):
        musat_ea_res = []
        p_sat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                            'Glaso': self.RS_glaso_cal,
                            'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                            'Lasater': self.RS_lasater_cal}
            pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                            'Glaso': self.Pb_glaso(),
                            'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                            'Lasater': self.Pb_lasater()}
            muod_dict_help = {'Standing': self.MUod_standing(), 'De_Ghetto': self.MUod_De_Ghetto(),
                              'Glaso': self.MUod_Glaso(), 'Elsharkawy_Alikhan': self.MUod_Elsharkawy_Alikhan(),
                              'Petorsky_Farshad': self.MUod_pf(), 'Beggs_Robinson': self.MUod_br(),
                              'Kartomodjo_Shmidt': self.MUod_krt_shdt()}
            if p <= pb_dict_help[rs_corr] * 14.5037:
                p_sat.append(p)
                rs = rs_dict_help[rs_corr](p)
                a = 0.2 + (0.8 / m.pow(10, 0.000852 * rs))
                b = 0.482 + (0.518 / m.pow(10, 0.000777 * rs))
                muob = a * m.pow(muod_dict_help[muod_corr], b)
                musat_ea_res.append(muob)

        df_mu_ea_res_sat = pd.DataFrame({'Pressure': p_sat, 'mu': musat_ea_res})
        df_mu_ea_res_sat['Pressure'] = df_mu_ea_res_sat['Pressure'] / 14.5037

        return df_mu_ea_res_sat, df_mu_ea_res_sat['mu'].iloc[-1]

    @functools.lru_cache
    def Musat_hossian(self,rs_corr,muod_corr): #некорректно работает
        musat_hos_res = []
        p_sat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                            'Glaso': self.RS_glaso_cal,
                            'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                            'Lasater': self.RS_lasater_cal}
            pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                            'Glaso': self.Pb_glaso(),
                            'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                            'Lasater': self.Pb_lasater()}
            muod_dict_help = {'Standing': self.MUod_standing(), 'De_Ghetto': self.MUod_De_Ghetto(),
                              'Glaso': self.MUod_Glaso(), 'Elsharkawy_Alikhan': self.MUod_Elsharkawy_Alikhan(),
                              'Petorsky_Farshad': self.MUod_pf(), 'Beggs_Robinson': self.MUod_br(),
                              'Kartomodjo_Shmidt': self.MUod_krt_shdt()}
            if p <= pb_dict_help[rs_corr] * 14.5037:
                p_sat.append(p)
                rs = rs_dict_help[rs_corr](p)
                a = 1 - 1.7188311 * m.pow(10,-3) * rs + 1.58031 * m.pow(10,-6) * m.pow(rs,2)
                b = 1 - 2.052461 * m.pow(10,-3) * rs + 3.47559 * m.pow(10,-6) * m.pow(rs,2)
                muob = a * m.pow(muod_dict_help[muod_corr], b)
                musat_hos_res.append(muob)

        df_mu_hos_res_sat = pd.DataFrame({'Pressure': p_sat, 'mu': musat_hos_res})
        df_mu_hos_res_sat['Pressure'] = df_mu_hos_res_sat['Pressure'] / 14.5037

        return df_mu_hos_res_sat, df_mu_hos_res_sat['mu'].iloc[-1]

    @functools.lru_cache
    def Musat_Petrosky_Farshad(self,rs_corr,muod_corr):
        musat_pf_res = []
        p_sat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                            'Glaso': self.RS_glaso_cal,
                            'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                            'Lasater': self.RS_lasater_cal}
            pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                            'Glaso': self.Pb_glaso(),
                            'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                            'Lasater': self.Pb_lasater()}
            muod_dict_help = {'Standing': self.MUod_standing(), 'De_Ghetto': self.MUod_De_Ghetto(),
                              'Glaso': self.MUod_Glaso(), 'Elsharkawy_Alikhan': self.MUod_Elsharkawy_Alikhan(),
                              'Petorsky_Farshad': self.MUod_pf(), 'Beggs_Robinson': self.MUod_br(),
                              'Kartomodjo_Shmidt': self.MUod_krt_shdt()}
            if p <= pb_dict_help[rs_corr] * 14.5037:
                p_sat.append(p)
                rs = rs_dict_help[rs_corr](p)
                a = 0.1651 + 0.6165 * m.pow(10, -6.0866 * m.pow(10,-4)*rs)
                b = 0.5131 + 0.5109 * m.pow(10, -1.1831 * m.pow(10,-3)*rs)
                muob = a * m.pow(muod_dict_help[muod_corr], b)
                musat_pf_res.append(muob)

        df_mu_pf_res_sat = pd.DataFrame({'Pressure': p_sat, 'mu': musat_pf_res})
        df_mu_pf_res_sat['Pressure'] = df_mu_pf_res_sat['Pressure'] / 14.5037

        return df_mu_pf_res_sat, df_mu_pf_res_sat['mu'].iloc[-1]

    @functools.lru_cache
    def Musat_Kartoatmodjo_Schmidt(self, rs_corr, muod_corr):
        musat_ks_res = []
        p_sat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        for p in p_massive:
            rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
                            'Glaso': self.RS_glaso_cal,
                            'Vasquez&Beggs': self.RS_vb_cal, 'Petorsky&Farshad': self.RS_pf_cal,
                            'Lasater': self.RS_lasater_cal}
            pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                            'Glaso': self.Pb_Glaso(),
                            'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                            'Lasater': self.Pb_lasater()}
            muod_dict_help = {'Standing': self.MUod_standing(), 'De_Ghetto': self.MUod_De_Ghetto(),
                              'Glaso': self.MUod_glaso(), 'Elsharkawy_Alikhan': self.MUod_Elsharkawy_Alikhan(),
                              'Petorsky_Farshad': self.MUod_pf(), 'Beggs_Robinson': self.MUod_br(),
                              'Kartomodjo_Shmidt': self.MUod_krt_shdt()}
            if p <= pb_dict_help[rs_corr] * 14.5037:
                p_sat.append(p)
                rs = rs_dict_help[rs_corr](p)
                a = 0.1651 + 0.6165 * m.pow(10, -6.0866 * m.pow(10, -4) * rs)
                b = 0.5131 + 0.5109 * m.pow(10, -1.1831 * m.pow(10, -3) * rs)
                muob = a * m.pow(muod_dict_help[muod_corr], b)
                musat_ks_res.append(muob)

        df_mu_ks_res_sat = pd.DataFrame({'Pressure': p_sat, 'mu': musat_ks_res})
        df_mu_ks_res_sat['Pressure'] = df_mu_ks_res_sat['Pressure'] / 14.5037

        return df_mu_ks_res_sat, df_mu_ks_res_sat['mu'].iloc[-1]

    '''
    ================== Вязкость нефти (недонасыщенная ветвь) ==================
    '''


    @functools.lru_cache
    def MUunsat_Standing(self, RS_corr, mu_sat_corr, muod_corr):
        munsat_standing_res = []
        p_unsat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                        'Glaso': self.Pb_glaso(),
                        'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                        'Lasater': self.Pb_lasater()}

        muob_dict_help = {'Standing': self.MUsat_Standing(RS_corr, muod_corr),
                          'Kartomodjo_Shmidt': self.MUsat_krt_shdt(RS_corr, muod_corr),
                          'Beggs_Robinson': self.Musat_beggs_robinson(RS_corr, muod_corr),
                          'Chew_Connanly': self.Musat_Chew_Connanly(RS_corr, muod_corr),
                          'Elsharkawy_Alikhan': self.Musat_Elsharkawy_Alikhan(RS_corr, muod_corr),
                          'Hossian': self.Musat_hossian(RS_corr, muod_corr),
                          'Petorsky_Farshad': self.Musat_Petrosky_Farshad(RS_corr, muod_corr)}
        pb = pb_dict_help[RS_corr] * 14.5037
        muob = muob_dict_help[mu_sat_corr][1]

        for p in p_massive:
            if p > pb:
                p_unsat.append(p)
                muou = muob + 0.001 * (p - pb) * (0.024 * muob ** 1.6 + 0.038 * muob ** 0.56)
                munsat_standing_res.append(muou)
        df_mu_standing_res_unsat = pd.DataFrame({'Pressure': p_unsat, 'mu': munsat_standing_res})
        df_mu_standing_res_unsat['Pressure'] = df_mu_standing_res_unsat['Pressure'] / 14.5037
        #plt.plot(df_mu_standing_res_unsat['Pressure'], munsat_standing_res)
        #plt.show()

        return df_mu_standing_res_unsat

    @functools.lru_cache
    def Muunsat_Vasquez_Beggs(self, RS_corr, mu_sat_corr, muod_corr):
        munsat_vb_res = []
        p_unsat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                        'Glaso': self.Pb_glaso(),
                        'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                        'Lasater': self.Pb_lasater()}

        muob_dict_help = {'Standing': self.MUsat_Standing(RS_corr, muod_corr),
                          'Kartomodjo_Shmidt': self.MUsat_krt_shdt(RS_corr, muod_corr),
                          'Beggs_Robinson': self.Musat_beggs_robinson(RS_corr, muod_corr),
                          'Chew_Connanly': self.Musat_Chew_Connanly(RS_corr, muod_corr),
                          'Elsharkawy_Alikhan': self.Musat_Elsharkawy_Alikhan(RS_corr, muod_corr),
                          'Hossian': self.Musat_hossian(RS_corr, muod_corr),
                          'Petorsky_Farshad': self.Musat_Petrosky_Farshad(RS_corr, muod_corr)}
        pb = pb_dict_help[RS_corr] * 14.5037
        muob = muob_dict_help[mu_sat_corr][1]

        for p in p_massive:
            if p > pb:
                p_unsat.append(p)
                a = 2.6 * m.pow(p,1.187) * m.exp(-11.513-8.98 * m.pow(10,-5) * p)
                muou = muob * m.pow((p/pb),a)
                munsat_vb_res.append(muou)
        df_mu_vb_res_unsat = pd.DataFrame({'Pressure': p_unsat, 'mu': munsat_vb_res})
        df_mu_vb_res_unsat['Pressure'] = df_mu_vb_res_unsat['Pressure'] / 14.5037
        # plt.plot(df_mu_standing_res_unsat['Pressure'], munsat_standing_res)
        # plt.show()

        return df_mu_vb_res_unsat

    @functools.lru_cache
    def Muunsat_Kouzel(self, RS_corr, mu_sat_corr, muod_corr): ## Не работает корректно
        munsat_kouzel_res = []
        p_unsat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                        'Glaso': self.Pb_glaso(),
                        'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                        'Lasater': self.Pb_lasater()}

        muob_dict_help = {'Standing': self.MUsat_Standing(RS_corr, muod_corr),
                          'Kartomodjo_Shmidt': self.MUsat_krt_shdt(RS_corr, muod_corr),
                          'Beggs_Robinson': self.Musat_beggs_robinson(RS_corr, muod_corr),
                          'Chew_Connanly': self.Musat_Chew_Connanly(RS_corr, muod_corr),
                          'Elsharkawy_Alikhan': self.Musat_Elsharkawy_Alikhan(RS_corr, muod_corr),
                          'Hossian': self.Musat_hossian(RS_corr, muod_corr),
                          'Petorsky_Farshad': self.Musat_Petrosky_Farshad(RS_corr, muod_corr)}
        muod_dict_help = {'Standing': self.MUod_standing(), 'De_Ghetto': self.MUod_De_Ghetto(),
                          'Glaso': self.MUod_Glaso(), 'Elsharkawy_Alikhan': self.MUod_Elsharkawy_Alikhan(),
                          'Petorsky_Farshad': self.MUod_pf(), 'Beggs_Robinson': self.MUod_br(),
                          'Kartomodjo_Shmidt': self.MUod_krt_shdt()}
        pb = pb_dict_help[RS_corr] * 14.5037
        muob = muob_dict_help[mu_sat_corr][1]

        for p in p_massive:
            if p > pb:
                p_unsat.append(p)
                a = 0.0239 ## параметры по умолчанию
                b = 0.01638
                muou = muob * m.pow(((p-pb) * (a + b * m.pow(muod_dict_help[muod_corr],0.278))/1000), 2)
                munsat_kouzel_res.append(muou)
        df_mu_kouzel_res_unsat = pd.DataFrame({'Pressure': p_unsat, 'mu': munsat_kouzel_res})
        df_mu_kouzel_res_unsat['Pressure'] = df_mu_kouzel_res_unsat['Pressure'] / 14.5037
        # plt.plot(df_mu_standing_res_unsat['Pressure'], munsat_standing_res)
        # plt.show()

        return df_mu_kouzel_res_unsat

    @functools.lru_cache
    def Muunsat_Petorsky_Farshad(self, RS_corr,mu_sat_corr, muod_corr):
        munsat_pf_res = []
        p_unsat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                        'Glaso': self.Pb_glaso(),
                        'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                        'Lasater': self.Pb_lasater()}

        muob_dict_help = {'Standing': self.MUsat_Standing(RS_corr, muod_corr),
                          'Kartomodjo_Shmidt': self.MUsat_krt_shdt(RS_corr, muod_corr),
                          'Beggs_Robinson': self.Musat_beggs_robinson(RS_corr, muod_corr),
                          'Chew_Connanly': self.Musat_Chew_Connanly(RS_corr, muod_corr),
                          'Elsharkawy_Alikhan': self.Musat_Elsharkawy_Alikhan(RS_corr, muod_corr),
                          'Hossian': self.Musat_hossian(RS_corr, muod_corr),
                          'Petorsky_Farshad': self.Musat_Petrosky_Farshad(RS_corr, muod_corr)}

        pb = pb_dict_help[RS_corr] * 14.5037
        muob = muob_dict_help[mu_sat_corr][1]

        for p in p_massive:
            if p > pb:
                p_unsat.append(p)
                x = m.log10(muob)
                a =  -1.0146 + 1.3322 * x - 0.4876 * m.pow(x,2) - 1.15036 * m.pow(x,3)
                muou = muob + 1.3449 * m.pow(10,-3) * (p - pb) * m.pow(10,a)
                munsat_pf_res.append(muou)
        df_mu_pf_res_unsat = pd.DataFrame({'Pressure': p_unsat, 'mu': munsat_pf_res})
        df_mu_pf_res_unsat['Pressure'] = df_mu_pf_res_unsat['Pressure'] / 14.5037
        # plt.plot(df_mu_standing_res_unsat['Pressure'], munsat_standing_res)
        # plt.show()

        return df_mu_pf_res_unsat

    @functools.lru_cache
    def Muunsat_Hossian(self, RS_corr, mu_sat_corr, muod_corr):
        munsat_hossian_res = []
        p_unsat = []
        p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
                              self.p_step_input_bar * 14.5037)
        pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
                        'Glaso': self.Pb_glaso(),
                        'Vasquez&Beggs': self.Pb_vb(), 'Petorsky&Farshad': self.Pb_pf(),
                        'Lasater': self.Pb_lasater()}

        muob_dict_help = {'Standing': self.MUsat_Standing(RS_corr, muod_corr),
                          'Kartomodjo_Shmidt': self.MUsat_krt_shdt(RS_corr, muod_corr),
                          'Beggs_Robinson': self.Musat_beggs_robinson(RS_corr, muod_corr),
                          'Chew_Connanly': self.Musat_Chew_Connanly(RS_corr, muod_corr),
                          'Elsharkawy_Alikhan': self.Musat_Elsharkawy_Alikhan(RS_corr, muod_corr),
                          'Hossian': self.Musat_hossian(RS_corr, muod_corr),
                          'Petorsky_Farshad': self.Musat_Petrosky_Farshad(RS_corr, muod_corr)}

        pb = pb_dict_help[RS_corr] * 14.5037
        muob = muob_dict_help[mu_sat_corr][1]

        for p in p_massive:
            if p > pb:
                p_unsat.append(p)
                a = 0.555955 * m.pow(muob,1.068099)
                b = 0.527737 * m.pow(muob,1.063547)
                muou = muob + 0.004481 * (p - pb) * (a-b)
                munsat_hossian_res.append(muou)
        df_mu_hossian_res_unsat = pd.DataFrame({'Pressure': p_unsat, 'mu': munsat_hossian_res})
        df_mu_hossian_res_unsat['Pressure'] = df_mu_hossian_res_unsat['Pressure'] / 14.5037
        # plt.plot(df_mu_standing_res_unsat['Pressure'], munsat_standing_res)
        # plt.show()

        return df_mu_hossian_res_unsat


    # def MUunsat_krt_shdt(self,rs_corr):
    #     rs_dict_help = {'Standing': self.RS_standing_cal, 'De_Ghetto': self.RS_deghetto_cal,
    #                     'Glaso': self.RS_glasso_cal,
    #                     'Vasquez_Beggs': self.RS_vb_cal, 'Petorsky_Farshad': self.RS_pf_cal,
    #                     'Lasater': self.RS_lasater_cal}
    #     pb_dict_help = {'Standing': self.Pb_standing(), 'De_Ghetto': self.Pb_deghetto(),
    #                     'Glaso': self.Pb_glaso(),
    #                     'Vasquez_Beggs': self.Pb_vb(), 'Petorsky_Farshad': self.Pb_pf(),
    #                     'Lasater': self.Pb_lasater(),}
    #
    #     p_massive = np.arange(self.p1_input_bar * 14.5037, self.p2_input_bar * 14.5037,
    #                           self.p_step_input_bar * 14.5037)
    #     mu_krt_shdt_res_unsat = []
    #     p_unsat = []
    #     pb = pb_dict_help[rs_corr] * 14.5037
    #     c = 16 * m.pow(10, 9) * m.pow(((self.t_input * 1.8) + 32), -2.8177)
    #     d = 5.7526 * m.log(((self.t_input * 1.8) + 32), 10) - 26.9718
    #     muod = c * m.pow(m.log(((141.5 / self.oden_input) - 131.5), 10), d)
    #     for p in p_massive:
    #         rs = rs_dict_help[rs_corr](p)
    #         if p > pb:
    #             p_unsat.append(p)
    #             a = 0.2001 + 0.8428 * m.pow(10, -8.45 * m.pow(10, -4) * self.rs_input_m3 * 35.314 / 6.289)
    #             f = a * m.pow(muod, 0.43 + 0.5165 * m.pow(10, - 8.1 * 10 ** -4 * self.rs_input_m3 * 35.314 / 6.289))
    #             muob = -0.06821 + 0.9824 * f + 4.034 * m.pow(10, -4) * m.pow(f, 2)
    #             a = -0.006517 * m.pow(muob, 1.8148) + 0.038 * m.pow(muob, 1.59)
    #             muou = 1.00081 * muob + 0.001127 * a * (p - pb)
    #             mu_krt_shdt_res_unsat.append(muou)
    #     df_mu_krt_shdt_res_unsat = pd.DataFrame({'Pressure': p_unsat, 'mu': mu_krt_shdt_res_unsat})
    #     df_mu_krt_shdt_res_unsat['Pressure'] = df_mu_krt_shdt_res_unsat['Pressure'] / 14.5037
    #     #plt.plot(df_mu_krt_shdt_res_unsat['Pressure'], mu_krt_shdt_res_unsat)
    #     #plt.show()
    #     return df_mu_krt_shdt_res_unsat


    @functools.lru_cache
    def mu_plot(self, rs_corr, muod_corr, mu_sat_corr, mu_unsat_corr):
        mu_sat_dict = {'Standing': self.MUsat_Standing(rs_corr, muod_corr),
                       'Kartomodjo_Shmidt': self.MUsat_krt_shdt(rs_corr, muod_corr),
                       'Beggs_Robinson':self.Musat_beggs_robinson(rs_corr,muod_corr),
                       'Chew_Connanly': self.Musat_Chew_Connanly(rs_corr, muod_corr),
                       'Elsharkawy_Alikhan': self.Musat_Elsharkawy_Alikhan(rs_corr, muod_corr),
                       'Hossian': self.Musat_hossian(rs_corr, muod_corr),
                       'Petorsky_Farshad': self.Musat_Petrosky_Farshad(rs_corr, muod_corr)}
        mu_unsat_dict = {'Standing': self.MUunsat_Standing(rs_corr,mu_sat_corr, muod_corr),
                         'Vasquez_Beggs': self.Muunsat_Vasquez_Beggs(rs_corr,mu_sat_corr, muod_corr),
                         'Kouzel': self.Muunsat_Kouzel(rs_corr,mu_sat_corr, muod_corr),
                         'Petorsky_Farshad': self.Muunsat_Petorsky_Farshad(rs_corr,mu_sat_corr, muod_corr),
                        'Hossian': self.Muunsat_Hossian(rs_corr,mu_sat_corr, muod_corr)}
                         #'Kartomodjo_Shmidt': self.MUunsat_krt_shdt(rs_corr)}
        mu_sat = mu_sat_dict[mu_sat_corr][0]
        mu_unsat = mu_unsat_dict[mu_unsat_corr]
        mu_res = pd.concat([mu_sat, mu_unsat], axis=0)
        plt.plot(mu_res['Pressure'], mu_res['mu'])
        plt.show()
        return mu_res







rs1 = PVT(rs_input_m3 = 120, oden_input= 0.86, t_input_c= 70, gden_input= 0.6, p1_input_bar=1, p2_input_bar=350,p_step_input_bar=1)
#print(rs1.MUunsat_standing('Standing'))
#rs1.mu_plot('Lasater','Petorsky_Farshad', 'Petorsky_Farshad', 'Hossian')
#rs1.cal_calc('Standing', 'Standing','Standing')
#rs1.MUunsat_standing('Standing','Standing')
#print(rs1.MUsat_standing('Standing')[1])

print(rs1.Bounsat_vb('Glaso', 'Vasquez_Beggs'))
