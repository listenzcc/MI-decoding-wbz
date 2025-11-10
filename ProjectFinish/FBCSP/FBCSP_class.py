

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.feature_selection import mutual_info_classif
import sklearn.feature_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import svm


class filter_bank():
    def __init__(self, freq_bands, fs, filt_order, filter_type='fir'):
        self.freq_bands = freq_bands
        self.fs = fs
        self.filt_order = filt_order
        self.filter_type = filter_type
        self.filters = []
        self.n_band = len(freq_bands)

        for pass_band in freq_bands:
            if filter_type == 'iir':
                b, a = signal.cheby1(N=filt_order, rp=1,
                                     Wn=pass_band, btype='bandpass', fs=fs)
            if filter_type == 'butter':
                b, a = signal.butter(filt_order, pass_band,
                                     fs=fs, btype="bandpass")
            elif filter_type == 'fir':
                b = signal.firwin(filt_order, pass_band,
                                  window='hamming', fs=fs, pass_zero='bandpass')
                a = 1
            self.filters.append([b, a])

    def filt(self, data):
        '''
        input data[epoch,channel,time]
        return filt_data[band,epoch,channel,time]
        '''
        filt_data = []
        for b, a in self.filters:
            y = signal.filtfilt(b, a, data, axis=-1,)  # padlen=self.fs
            filt_data.append(y)
        filt_data = np.asarray(filt_data)

        return filt_data

    def plot_filter(self, range=(0, 60)):
        import matplotlib.pyplot as plt
        import numpy as np
        fs = self.fs
        for b, a in self.filters:
            w, h = signal.freqz(b, a)
            fig, ax1 = plt.subplots()
            ax1.set_title('Digital filter frequency response')

            ax1.plot(w/np.pi*fs/2, 20 * np.log10(abs(h)), 'b')
            ax1.set_ylabel('Amplitude [dB]', color='b')
            ax1.set_xlabel('Frequency [rad/sample]')
            plt.xlim(0, 60)
            plt.ylim(-60, 0)

            ax2 = ax1.twinx()
            angles = np.unwrap(np.angle(h))
            ax2.plot(w/np.pi*fs/2, angles, 'g')
            ax2.set_ylabel('Angle (radians)', color='g')
            ax2.grid()
            ax2.axis('tight')

            plt.xlim(range)
            plt.show()


class FBCSP():
    def __init__(self, FB, n_components_list, crop_list):
        self.n_components_list = n_components_list
        fs = FB.fs
        self.crop_list = [
            [int((crop_list[i][0])*fs),
             int((crop_list[i][1])*fs)]
            for i in range(len(crop_list))
        ]

        self.FB = FB
        self.n_band = FB.n_band

    def fit(self, train_data, train_labels):

        self.min_label = np.min(train_labels)

        # filter bank
        fb_data_trials = self.FB.filt(train_data)
        feature_bands = []

        # train csp and clf for each band
        csp_bands = []
        clf_bands = []
        for i_freq in range(self.n_band):
            # print('1')
            n_components = self.n_components_list[i_freq]
            crop = self.crop_list[i_freq]
            # print(crop)
            X_train_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            # print(X_train_filt.shape)
            csp = CSP(n_components=n_components, reg=None,
                      cov_est='concat', log=True, norm_trace=False,)

            X_feature = csp.fit_transform(X_train_filt, train_labels)
            csp_bands.append(csp)

            clf = LinearDiscriminantAnalysis(solver='lsqr')
            clf.fit(X_feature, train_labels)
            clf_bands.append(clf)

        self.csp_bands = csp_bands
        self.clf_bands = clf_bands

    def fit_stacking(self, train_data, train_labels):
        self.fit(train_data, train_labels)
        # stacking train
        stacking_feature = np.array(self.prob(train_data)).swapaxes(0, 1).\
            reshape((train_data.shape[0], -1))

        clf = LogisticRegression(multi_class='auto', solver='lbfgs').\
            fit(stacking_feature, train_labels)

        self.stacking_clf = clf

    def predict(self, test_data):
        feature_bands = []
        predict_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop_list[i_freq]
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            predict_bands.append(
                self.clf_bands[i_freq].predict(X_feature)
            )

        return predict_bands

    def prob(self, test_data):
        feature_bands = []
        prob_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop_list[i_freq]
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            prob_bands.append(
                self.clf_bands[i_freq].predict_proba(X_feature)
            )

        return prob_bands

    def predict_all(self, test_data):
        predict_bands = self.predict(test_data)
        hard_pred = hard_vote(predict_bands)

        # shape is (n_bands, n_samples, n_classes)
        # predict_bands_prob = self.prob(test_data)
        # print(f'{predict_bands_prob=}, {np.array(predict_bands_prob).shape=}')
        # soft_pred = soft_vote(predict_bands_prob)
        # print(f'{hard_pred.shape=}')
        # print(f'{soft_pred.shape=}')
        # stophere

        prob_bands = self.prob(test_data)
        soft_pred = np.argmax(np.mean(prob_bands, axis=0),
                              axis=-1)+self.min_label

        test_feature = np.array(prob_bands).swapaxes(
            0, 1).reshape((test_data.shape[0], -1))

        # stacking_predict = self.stacking_clf.predict(test_feature)
        stacking_predict = None

        return predict_bands, hard_pred, soft_pred, stacking_predict


class FBCSP_info():
    def __init__(self, FB, n_components, crop, k_select):

        self.n_components = n_components
        fs = FB.fs
        self.crop = [
            int((crop[0])*fs),
            int((crop[1])*fs)]

        self.FB = FB
        self.n_band = FB.n_band
        self.k_select = k_select

    def fit(self, train_data, train_labels,):

        self.min_label = np.min(train_labels)
        # filter bank
        fb_data_trials = self.FB.filt(train_data)

        feature_bands = []
        # train csp and clf for each band
        csp_bands = []

        for i_freq in range(self.n_band):
            n_components = self.n_components
            crop = self.crop
            X_train_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = CSP(n_components=n_components, reg=None,
                      cov_est='concat', log=True, norm_trace=False,)

            X_feature = csp.fit_transform(X_train_filt, train_labels)
            csp_bands.append(csp)

            feature_bands.append(X_feature)

        # info base feature selection
        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))

        select_K = sklearn.feature_selection.SelectKBest(mutual_info_classif,
                                                         k=self.k_select,).fit(feature_bands, train_labels)

        feature_selected = select_K.transform(feature_bands)

        clf = LinearDiscriminantAnalysis(solver='lsqr')  # solver='eigen'
        clf.fit(feature_selected, train_labels)

        # save trained models
        self.select_K = select_K
        self.csp_bands = csp_bands
        self.clf = clf

    def predict(self, test_data):
        feature_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            feature_bands.append(X_feature)

        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))
        feature_selected = self.select_K.transform(feature_bands)
        predict = self.clf.predict(feature_selected)

        return predict

    def decision_function(self, test_data):
        feature_selected = self.extract_f(test_data)
        decision_value = self.clf.decision_function(feature_selected)
        return decision_value

    def extract_f(self, test_data):
        feature_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            feature_bands.append(X_feature)

        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))
        feature_selected = self.select_K.transform(feature_bands)
        return feature_selected

    def prob(self, test_data):
        feature_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            feature_bands.append(X_feature)

        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))
        feature_selected = self.select_K.transform(feature_bands)
        prob = self.clf.predict_proba(feature_selected)

        return prob

    def transform(self, test_data):
        feature_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            feature_bands.append(X_feature)

        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))
        feature_selected = self.select_K.transform(feature_bands)

        return feature_selected


class FBCSP_info_weighted():
    def __init__(self, FB, n_components, crop, k_select):

        self.n_components = n_components
        fs = FB.fs
        self.crop = [
            int((crop[0])*fs),
            int((crop[1])*fs)]

        self.FB = FB
        self.n_band = FB.n_band
        self.k_select = k_select

    def fit(self, train_data, train_labels, weights):

        self.min_label = np.min(train_labels)
        # filter bank
        fb_data_trials = self.FB.filt(train_data)

        feature_bands = []
        # train csp and clf for each band
        csp_bands = []

        for i_freq in range(self.n_band):
            n_components = self.n_components
            crop = self.crop
            X_train_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = CSP(n_components=n_components, reg=None,
                      cov_est='concat', log=True, norm_trace=False,)

            # X_feature=csp.fit_transform(X_train_filt,train_labels)
            csp.fit(X_train_filt*weights, train_labels)
            X_feature = csp.transform(X_train_filt)

            csp_bands.append(csp)

            feature_bands.append(X_feature)

        # info base feature selection
        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))

        select_K = sklearn.feature_selection.SelectKBest(mutual_info_classif,
                                                         k=self.k_select,).fit(feature_bands, train_labels)

        feature_selected = select_K.transform(feature_bands)

        clf = LinearDiscriminantAnalysis(solver='lsqr')  # solver='eigen'
        clf.fit(feature_selected, train_labels)

        # save trained models
        self.select_K = select_K
        self.csp_bands = csp_bands
        self.clf = clf

    def predict(self, test_data):
        feature_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            feature_bands.append(X_feature)

        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))
        feature_selected = self.select_K.transform(feature_bands)
        predict = self.clf.predict(feature_selected)

        return predict

    def decision_function(self, test_data):
        feature_selected = self.extract_f(test_data)
        decision_value = self.clf.decision_function(feature_selected)
        return decision_value

    def extract_f(self, test_data):
        feature_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            feature_bands.append(X_feature)

        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))
        feature_selected = self.select_K.transform(feature_bands)
        return feature_selected

    def prob(self, test_data):
        feature_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            feature_bands.append(X_feature)

        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))
        feature_selected = self.select_K.transform(feature_bands)
        prob = self.clf.predict_proba(feature_selected)

        return prob

    def transform(self, test_data):
        feature_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            feature_bands.append(X_feature)

        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))
        feature_selected = self.select_K.transform(feature_bands)

        return feature_selected


def hard_vote(multi_preds):
    # multi_preds[model,sample]
    multi_preds = np.array(multi_preds)
    vote = []
    for i_sample in range(multi_preds.shape[1]):
        vote.append(np.argmax(np.bincount(multi_preds[:, i_sample])))
    vote = np.array(vote)
    return vote

def soft_vote(multi_preds):
    # multi_preds shape is (n_bands, n_samples, n_classes)
    multi_preds = np.array(multi_preds)
    joint_preds = np.prod(multi_preds, axis=0)
    vote = np.argmax(joint_preds, axis=1)
    # Convert into real labels
    vote = vote + 1
    return vote



def crop_data(train_data, train_labels, slice_crop, fs):
    train_data_all = []
    train_labels_all = []
    for i_slice in slice_crop:
        retaled_crop = [
            int((i_slice[0])*fs),
            int((i_slice[1])*fs)
        ]
        train_data_all.append(
            train_data[:, :, retaled_crop[0]:retaled_crop[1]])
        train_labels_all.append(train_labels)
    train_data_all = np.array(train_data_all)
    d1, d2, d3, d4 = train_data_all.shape
    train_data = train_data_all.reshape(d1*d2, d3, d4)
    train_labels = np.array(train_labels_all).reshape(-1)

    return train_data, train_labels


class FBCSP_C2CM():
    def __init__(self, FB, n_components, crop, k_select):

        self.n_components = n_components
        fs = FB.fs
        self.crop = [
            int((crop[0])*fs),
            int((crop[1])*fs)]

        self.FB = FB
        self.n_band = FB.n_band
        self.k_select = k_select

    def fit(self, train_data, train_labels,):

        self.min_label = np.min(train_labels)
        # filter bank
        fb_data_trials = self.FB.filt(train_data)

        feature_bands = []
        # train csp and clf for each band
        csp_bands = []

        for i_freq in range(self.n_band):
            n_components = self.n_components
            crop = self.crop
            X_train_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = CSP(n_components=n_components, reg=None,
                      cov_est='concat', log=True, norm_trace=False,)

            X_feature = csp.fit_transform(X_train_filt, train_labels)
            csp_bands.append(csp)

            feature_bands.append(X_feature)

        # info base feature selection
        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))

        select_K = sklearn.feature_selection.SelectKBest(mutual_info_classif,
                                                         k=self.k_select,).fit(feature_bands, train_labels)

        feature_selected = select_K.transform(feature_bands)

        clf = LinearDiscriminantAnalysis(solver='lsqr')  # solver='eigen'
        clf.fit(feature_selected, train_labels)

        # save trained models
        self.select_K = select_K
        self.csp_bands = csp_bands
        self.clf = clf

    def predict(self, test_data):
        feature_bands = []
        for i_freq in range(self.n_band):
            fb_data_trials = self.FB.filt(test_data)
            crop = self.crop
            X_filt = fb_data_trials[i_freq, :, :, crop[0]:crop[1]]
            csp = self.csp_bands[i_freq]
            X_feature = csp.transform(X_filt)
            feature_bands.append(X_feature)

        feature_bands = np.array(feature_bands)
        feature_bands = feature_bands.transpose(1, 0, 2).reshape(
            (-1, int(self.n_band*self.n_components)))
        feature_selected = self.select_K.transform(feature_bands)
        predict = self.clf.predict(feature_selected)

        return predict


def hard_vote_old(multi_preds):
    # multi_preds[model,sample]
    multi_preds = np.array(multi_preds)
    vote = []
    for i_sample in range(multi_preds.shape[1]):
        vote.append(np.argmax(np.bincount(multi_preds[:, i_sample])))
    vote = np.array(vote)
    return vote


def crop_data_old(train_data, train_labels, slice_crop, fs):
    train_data_all = []
    train_labels_all = []
    for i_slice in slice_crop:
        retaled_crop = [
            int((i_slice[0])*fs),
            int((i_slice[1])*fs)
        ]
        train_data_all.append(
            train_data[:, :, retaled_crop[0]:retaled_crop[1]])
        train_labels_all.append(train_labels)
    train_data_all = np.array(train_data_all)
    d1, d2, d3, d4 = train_data_all.shape
    train_data = train_data_all.reshape(d1*d2, d3, d4)
    train_labels = np.array(train_labels_all).reshape(-1)

    return train_data, train_labels


class CSP_MM():
    def __init__(self, FB_EEG, n_components):
        self.n_components = n_components
        self.FB_EEG = FB_EEG
        self.n_band = FB_EEG.n_band

    def fit(self, x_EEG, x_NIR, y):

        # filter bank
        fb_x_EEG = self.FB_EEG.filt(x_EEG)
        fb_x_NIR = x_NIR  # self.FB_NIR.filt(x_NIR)

        # train csp and clf for each band
        csp_bands_EEG = []
        csp_bands_NIR = []
        clf_bands = []
        for i_freq in range(self.n_band):
            # print('1')
            n_components = self.n_components

            # print(crop)
            band_x_EEG = fb_x_EEG[i_freq, :, :, :]
            band_x_NIR = x_NIR
            # print(X_train_filt.shape)
            csp_EEG = CSP(n_components=n_components, reg=None,
                          cov_est='concat', log=True, norm_trace=False,)
            csp_NIR = CSP(n_components=n_components, reg=None,
                          cov_est='concat', log=True, norm_trace=False,)

            band_f_EEG = csp_EEG.fit_transform(band_x_EEG, y)
            csp_bands_EEG.append(csp_EEG)

            band_f_NIR = csp_NIR.fit_transform(band_x_NIR, y)
            csp_bands_NIR.append(csp_NIR)

            band_f_MM = np.concatenate((band_f_EEG, band_f_NIR), axis=-1)

            clf = svm.SVC(C=1, kernel='rbf', decision_function_shape='ovr')
            clf.fit(band_f_MM, y)
            clf_bands.append(clf)

        self.csp_bands_EEG = csp_bands_EEG
        self.csp_bands_NIR = csp_bands_NIR
        self.clf_bands = clf_bands

    def predict(self, x_EEG, x_NIR):
        # filter bank
        fb_x_EEG = self.FB_EEG.filt(x_EEG)
        fb_x_NIR = x_NIR
        predict_bands = []
        for i_freq in range(self.n_band):
            band_x_EEG = fb_x_EEG[i_freq, :, :, :]
            band_x_NIR = fb_x_NIR

            csp_EEG = self.csp_bands_EEG[i_freq]
            csp_NIR = self.csp_bands_NIR[i_freq]
            band_f_EEG = csp_EEG.transform(band_x_EEG)
            band_f_NIR = csp_NIR.transform(band_x_NIR)
            band_f_MM = np.concatenate((band_f_EEG, band_f_NIR), axis=-1)

            predict_bands.append(
                self.clf_bands[i_freq].predict(band_f_MM)
            )

        return predict_bands


class FBCSP_MM():
    def __init__(self, FB_EEG, n_components):
        self.n_components = n_components
        self.FB_EEG = FB_EEG
        self.n_band = FB_EEG.n_band

    def fit(self, x_EEG, x_NIR, y):

        # filter bank
        fb_x_EEG = self.FB_EEG.filt(x_EEG)
        fb_x_NIR = x_NIR  # self.FB_NIR.filt(x_NIR)

        # train csp and clf for each band
        csp_bands_EEG = []
        csp_bands_NIR = []
        clf_bands = []
        band_feature = []
        for i_freq in range(self.n_band):
            # print('1')
            n_components = self.n_components

            # print(crop)
            band_x_EEG = fb_x_EEG[i_freq, :, :, :]
            band_x_NIR = x_NIR
            # print(X_train_filt.shape)
            csp_EEG = CSP(n_components=n_components, reg=None,
                          cov_est='concat', log=True, norm_trace=False,)

            band_f_EEG = csp_EEG.fit_transform(band_x_EEG, y)
            csp_bands_EEG.append(csp_EEG)

            band_feature.append(band_f_EEG)

        csp_NIR = CSP(n_components=n_components, reg=None,
                      cov_est='concat', log=True, norm_trace=False,)
        band_f_NIR = csp_NIR.fit_transform(band_x_NIR, y)
        csp_bands_NIR.append(csp_NIR)
        # band_feature.append(band_f_NIR)

        band_f_MM = np.concatenate(band_feature, axis=-1)

        clf = svm.SVC(C=1, kernel='rbf', decision_function_shape='ovr')
        clf.fit(band_f_MM, y)
        clf_bands.append(clf)

        self.csp_bands_EEG = csp_bands_EEG
        self.csp_bands_NIR = csp_bands_NIR
        self.clf_bands = clf_bands

    def predict(self, x_EEG, x_NIR):
        # filter bank
        fb_x_EEG = self.FB_EEG.filt(x_EEG)
        fb_x_NIR = x_NIR
        predict_bands = []
        band_feature = []
        for i_freq in range(self.n_band):
            band_x_EEG = fb_x_EEG[i_freq, :, :, :]
            band_x_NIR = fb_x_NIR

            csp_EEG = self.csp_bands_EEG[i_freq]
            band_f_EEG = csp_EEG.transform(band_x_EEG)
            band_feature.append(band_f_EEG)

        csp_NIR = self.csp_bands_NIR[0]
        band_f_NIR = csp_NIR.transform(band_x_NIR)
        # band_feature.append(band_f_NIR)

        band_f_MM = np.concatenate(band_feature, axis=-1)

        predict_bands = self.clf_bands[0].predict(band_f_MM)

        return predict_bands
