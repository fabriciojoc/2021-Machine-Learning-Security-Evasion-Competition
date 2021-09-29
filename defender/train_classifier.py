import os
import gzip
import json
import pickle
import numpy as np
import pandas as pd
# imports
import _pickle as cPickle
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from copy import deepcopy

def load_gzip_pickle(filename):
    fp = gzip.open(filename,'rb')
    obj = cPickle.load(fp)
    fp.close()
    return obj


def save_gzip_pickle(filename, obj):
    fp = gzip.open(filename,'wb')
    cPickle.dump(obj,fp)
    fp.close()


class JSONAttributeExtractor():

    # initialize extractor
    def __init__(self, file):
        # save data
        self.data = json.loads(file)
        # attributes
        self.attributes = {}

    # extract string metadata
    def extract_string_metadata(self):
        return {
            'string_paths': self.data["strings"]["paths"],
            'string_urls': self.data["strings"]["urls"],
            'string_registry': self.data["strings"]["registry"],
            'string_MZ': self.data["strings"]["MZ"]
        }

    # extract attributes
    def extract(self):

        # get general info
        self.attributes.update({
            "size": self.data["general"]["size"], 
            "virtual_size": self.data["general"]["vsize"],
            "has_debug": self.data["general"]["has_debug"], 
            "imports": self.data["general"]["imports"],
            "exports": self.data["general"]["exports"],
            "has_relocations": self.data["general"]["has_relocations"],
            "has_resources": self.data["general"]["has_resources"],
            "has_signature": self.data["general"]["has_signature"],
            "has_tls": self.data["general"]["has_tls"],
            "symbols": self.data["general"]["symbols"],
        })

        # get header info
        self.attributes.update({
            "timestamp": self.data["header"]["coff"]["timestamp"],
            # NOTE: Machine is a string, we need to transform it in a categorical feature
            # https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
            "machine": self.data["header"]["coff"]["machine"],
            # TODO: NFS only
            "numberof_sections": len(self.data["section"]["sections"]),
            "characteristics_list": " ".join(self.data["header"]["coff"]["characteristics"])
        })

       # get optional header
        self.attributes.update({
            "dll_characteristics_list": " ".join(self.data["header"]["optional"]["dll_characteristics"]),
            "magic": self.data["header"]["optional"]["magic"],
            # EMBER only
            "major_image_version": self.data["header"]["optional"]["major_image_version"],
            # EMBER only
            "minor_image_version": self.data["header"]["optional"]["minor_image_version"],
            # EMBER only
            "major_linker_version": self.data["header"]["optional"]["major_linker_version"],
            # EMBER only
            "minor_linker_version": self.data["header"]["optional"]["minor_linker_version"],
            # EMBER only
            "major_operating_system_version": self.data["header"]["optional"]["major_operating_system_version"],
            # EMBER only
            "minor_operating_system_version": self.data["header"]["optional"]["minor_operating_system_version"],
            # EMBER only
            "major_subsystem_version": self.data["header"]["optional"]["major_subsystem_version"],
            # EMBER only
            "minor_subsystem_version": self.data["header"]["optional"]["minor_subsystem_version"],
            "sizeof_code": self.data["header"]["optional"]["sizeof_code"],
            "sizeof_headers": self.data["header"]["optional"]["sizeof_headers"],
            # EMBER only
            "sizeof_heap_commit": self.data["header"]["optional"]["sizeof_heap_commit"]
        })

        # get string metadata
        # EMBER only
        self.attributes.update(self.extract_string_metadata())

        # get imported libraries and functions
        self.libraries = " ".join([item for sublist in self.data["imports"].values() for item in sublist])
        self.libraries = " {} ".format(self.libraries)
        self.functions = " ".join(self.data["imports"].keys())
        self.functions = " {} ".format(self.functions)
        self.attributes.update({"functions": self.functions, "libraries": self.libraries})

        # get exports
        self.exports = " ".join(self.data["exports"])
        self.attributes.update({"exports_list": self.exports})

        # get label
        self.label = self.data["label"]
        self.attributes.update({"label": self.label})

        return(self.attributes)

# need for speed class
class NeedForSpeedModel():

    # numerical attributes
    NUMERICAL_ATTRIBUTES = [
        #'string_paths', 'string_urls', 'string_registry', 'string_MZ', 'size',
        'virtual_size', 'has_debug', 'imports', 'exports', 'has_relocations',
        'has_resources', 'has_signature', 'has_tls', 'symbols', 'timestamp', 
        'numberof_sections', 'major_image_version', 'minor_image_version', 
        'major_linker_version', 'minor_linker_version', 'major_operating_system_version',
        'minor_operating_system_version', 'major_subsystem_version', 
        'minor_subsystem_version', 'sizeof_code', 'sizeof_headers', 'sizeof_heap_commit'
    ]

    # categorical attributes
    CATEGORICAL_ATTRIBUTES = [
        'machine', 'magic'
    ]

    # textual attributes
    TEXTUAL_ATTRIBUTES = ['libraries', 'functions', 'exports_list',
                          'dll_characteristics_list', 'characteristics_list']

    #'dll_characteristics_list' and 'characteristics_list' are texts or multi-categoricals??

    # label
    LABEL = "label"

    # initialize NFS classifier
    def __init__(self, 
                categorical_extractor = OneHotEncoder(handle_unknown="ignore"), 
                # textual_extractor = TfidfVectorizer(max_features=500, token_pattern=r"(?<=\s)(.*?)(?=\s)"),
                textual_extractor = HashingVectorizer(n_features=50000, token_pattern=r"(?<=\s)(.*?)(?=\s)"),
                #feature_scaler = MinMaxScaler(),
                feature_scaler = MaxAbsScaler(),
                classifier = RandomForestClassifier()):
        self.base_categorical_extractor = categorical_extractor
        self.base_textual_extractor = textual_extractor
        self.base_feature_scaler = feature_scaler
        self.base_classifier = classifier

    # append features to original features list
    def _append_features(self, original_features, appended):
        if original_features:
            for l1, l2 in zip(original_features, appended):
                for i in l2:
                    l1.append(i)
            return(original_features)
        else:
            return appended.tolist()

    # train a categorical extractor
    def _train_categorical_extractor(self, categorical_attributes):
        # initialize categorical extractor
        self.categorical_extractor = deepcopy(self.base_categorical_extractor)
        # train categorical extractor
        self.categorical_extractor.fit(categorical_attributes.values)

    # transform categorical attributes into features
    def _transform_categorical_attributes(self, categorical_attributes):
        # transform categorical attributes using categorical extractor
        cat_features = self.categorical_extractor.transform(categorical_attributes.values)
        # return categorical features
        return cat_features

    # train a textual extractor
    def _train_textual_extractor(self, textual_attributes):
        # initialize textual extractors
        self.textual_extractors = {}
        # train feature extractor for each textual attribute
        for att in self.TEXTUAL_ATTRIBUTES:
            # initialize textual extractors
            self.textual_extractors[att] = deepcopy(self.base_textual_extractor)
            # train textual extractor
            self.textual_extractors[att].fit(textual_attributes[att].values)
    
    # transform textual extractor
    def _transform_textual_attributes(self, textual_attributes):
        # initialize features
        textual_features = None
        # extract features from each textual attribute
        for att in self.TEXTUAL_ATTRIBUTES:
            # train textual extractor
            att_features = self.textual_extractors[att].transform(textual_attributes[att].values)
            # transform into array (when it is an sparse matrix)
            # att_features = att_features.toarray()
            if textual_features == None:
                textual_features = att_features
            else:
                # append textual features
                textual_features = sparse.hstack((textual_features, att_features))
            # append textual features
            # textual_features = self._append_features(textual_features, att_features)
        return textual_features
        
    # train feature scaler
    def _train_feature_scaler(self, features):
        # initialize feature scaler
        self.feature_scaler = deepcopy(self.base_feature_scaler)
        # train feature scaler
        self.feature_scaler.fit(features)

    # transform features using feature scaler
    def _transform_feature_scaler(self, features):
        return self.feature_scaler.transform(features)

    # train classifier
    def _train_classifier(self,features,labels):
        # initialize classifier
        self.classifier = deepcopy(self.base_classifier)
        # train feature scaler
        self.classifier.fit(features, labels)

    # fit classifier using raw input
    def fit(self, train_data):
        # get labels
        train_labels = train_data[self.LABEL]
        # delete label column
        del train_data[self.LABEL]
        # initialize train_features with numerical ones
        train_features = sparse.csr_matrix(train_data[self.NUMERICAL_ATTRIBUTES].values)

        print("Training categorical features...", flush=True)
        # train categorical extractor
        self._train_categorical_extractor(train_data[self.CATEGORICAL_ATTRIBUTES])
        # transform categorical data
        cat_train_features = self._transform_categorical_attributes(train_data[self.CATEGORICAL_ATTRIBUTES])
        # append categorical_features to train_features
        # train_features = self._append_features(train_features, cat_train_features)
        train_features = sparse.hstack((train_features, cat_train_features))

        print("Training textual features...", flush=True)
        # train textual extractor (ALL DATA)
        self._train_textual_extractor(train_data[self.TEXTUAL_ATTRIBUTES])
        # train textual extractor (MALWARE ONLY)
        # self._train_textual_extractor(train_data[train_labels == 1][self.TEXTUAL_ATTRIBUTES])
        # transform textual data
        tex_train_features = self._transform_textual_attributes(train_data[self.TEXTUAL_ATTRIBUTES])
        # append textual_features to train_features
        # train_features = self._append_features(train_features, tex_train_features)
        train_features = sparse.hstack((train_features, tex_train_features))
        # transform in sparse matrix
        # train_features = csr_matrix(train_features)

        print("Normalizing features...", flush=True)
        # train feature normalizer
        self._train_feature_scaler(train_features)
        # transform features
        train_features = self._transform_feature_scaler(train_features)

        print("Training classifier...", flush=True)
        # train classifier
        return self._train_classifier(train_features, train_labels)


    def _extract_features(self,data):
        # initialize features with numerical ones
        # features = data[self.NUMERICAL_ATTRIBUTES].values.tolist()
        features = sparse.csr_matrix(data[self.NUMERICAL_ATTRIBUTES].values)

        print("Getting categorical features...", flush=True)
        # transform categorical data
        cat_features = self._transform_categorical_attributes(data[self.CATEGORICAL_ATTRIBUTES])
        # append categorical_features to features
        # features = self._append_features(features, cat_features)
        features = sparse.hstack((features, cat_features))

        print("Getting textual features...", flush=True)
        # transform textual data
        tex_features = self._transform_textual_attributes(data[self.TEXTUAL_ATTRIBUTES])
        # append textual_features to features
        # features = self._append_features(features, tex_features)
        features = sparse.hstack((features, tex_features))
        # transform in sparse matrix
        # features = csr_matrix(features)

        print("Normalizing features...", flush=True)
        # transform features
        features = self._transform_feature_scaler(features)

        # return features
        return(features)

    def predict(self,test_data):
        # extract features
        test_features = self._extract_features(test_data)        

        print("Predicting classes...", flush=True)
        # predict features
        return self.classifier.predict(test_features)

    def predict_proba(self,test_data):
        # extract features
        test_features = self._extract_features(test_data)        

        print("Predicting classes (proba)...", flush=True)
        # predict features
        return self.classifier.predict_proba(test_features)

    def predict_threshold(self,test_data, threshold=0.75):
        # extract features
        test_features = self._extract_features(test_data)        

        print("Predicting classes (threshold = {})...".format(threshold), flush=True)
        # predict features
        prob = self.classifier.predict_proba(test_features)
        # initialize pred
        pred = []
        # iterate over probabilities
        for p in prob:
            # add prediction
            pred.append(int(p[0] < threshold))
        # return prediction
        return pred


THRESHOLD = 0.75
CLF_FILE = "NFS_21_ALL_hash_50000_WITH_MLSEC20.pkl"

train_files = [
    "/home/fabricioceschin/ember/ember/train_features_0.jsonl.gzip",
    "/home/fabricioceschin/ember/ember/train_features_1.jsonl.gzip",
    "/home/fabricioceschin/ember/ember/train_features_2.jsonl.gzip",
    "/home/fabricioceschin/ember/ember/train_features_3.jsonl.gzip",
    "/home/fabricioceschin/ember/ember/train_features_4.jsonl.gzip",
    "/home/fabricioceschin/ember/ember/train_features_5.jsonl.gzip",
    "/home/fabricioceschin/ember/ember_2017_2/train_features_0.jsonl.gzip",
    "/home/fabricioceschin/ember/ember_2017_2/train_features_1.jsonl.gzip",
    "/home/fabricioceschin/ember/ember_2017_2/train_features_2.jsonl.gzip",
    "/home/fabricioceschin/ember/ember_2017_2/train_features_3.jsonl.gzip",
    "/home/fabricioceschin/ember/ember_2017_2/train_features_4.jsonl.gzip",
    "/home/fabricioceschin/ember/ember_2017_2/train_features_5.jsonl.gzip",
    "/home/fabricioceschin/ember/ember2018/train_features_0.jsonl.gzip",
    "/home/fabricioceschin/ember/ember2018/train_features_1.jsonl.gzip",
    "/home/fabricioceschin/ember/ember2018/train_features_2.jsonl.gzip",
    "/home/fabricioceschin/ember/ember2018/train_features_3.jsonl.gzip",
    "/home/fabricioceschin/ember/ember2018/train_features_4.jsonl.gzip",
    "/home/fabricioceschin/ember/ember2018/train_features_5.jsonl.gzip",
]


test_files = [
    "/home/fabricioceschin/ember/ember/test_features.jsonl.gzip",
    "/home/fabricioceschin/ember/ember_2017_2/test_features.jsonl.gzip",
    "/home/fabricioceschin/ember/ember2018/test_features.jsonl.gzip"
]

adv_files = [
    "/home/fabricioceschin/ember/adversaries/mlsec19.jsonl",
    "/home/fabricioceschin/ember/adversaries/mlsec20.jsonl",
]

if __name__=='__main__':

    if not os.path.isfile(CLF_FILE):
        train_attributes = []
        gw_data = []
        mw_data = []
        # walk in train features
        for input in train_files:
            
            print("Reading {}...".format(input), flush=True)

            # read input file
            if 'mlsec' in input or 'UCSB' in input:
                file = open(input, 'r') 
            else:
                file = gzip.open(input, 'rb')
            # read its lines
            sws = file.readlines() 
            # print(len(sws))
            
            # walk in each sw
            for sw in sws: 
                if 'mlsec' in input or 'UCSB' in input:
                    # atts = at_extractor.extract()
                    atts = json.loads(sw)
                    # print( == 0)

                    # if 'UCSB_gw' in input:
                    #     imbalance_count +=1
                    #     if imbalance_count <= 1477:                        
                    #         train_attributes.append(atts)
                    # else:
                    #     train_attributes.append(atts)
                    # print(atts)
                else:
                    # initialize extractor
                    at_extractor = JSONAttributeExtractor(sw)
                    # get train_attributes
                    atts = at_extractor.extract()
                # save attribute
                train_attributes.append(atts)

            # close file
            file.close()
        # transform into pandas dataframe
        train_data = pd.DataFrame(train_attributes)
        # create a NFS model        
        clf = NeedForSpeedModel(classifier=RandomForestClassifier(n_jobs=-1))
        # train it
        clf.fit(train_data)
        # save clf
        print("Saving model...", flush=True)
        # save it
        save_gzip_pickle(CLF_FILE, clf)
    else:
        # model already trained, use it to test
        print("Loading saved classifer...")
        # load model
        clf = load_gzip_pickle(CLF_FILE)
    
    test_attributes = []
    # walk in test features
    for input in test_files:
        
        print("Reading {}...".format(input))

        # read input file
        # file = open(input, 'r') 
        file = gzip.open(input, 'rb')
        # read its lines
        sws = file.readlines() 
        
        # walk in each sw
        for sw in sws: 
            # initialize extractor
            at_extractor = JSONAttributeExtractor(sw)
            # get test_attributes
            atts = at_extractor.extract()
            # save attribute
            test_attributes.append(atts)

        # close file
        file.close()

    test_data = pd.DataFrame(test_attributes)
    test_data = test_data[(test_data["label"]==1) | (test_data["label"]==0)]
    #print(test_data)
    print(test_data.shape)

    test_label = test_data["label"].values
    y_pred = clf.predict(test_data)

    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    from sklearn.metrics import confusion_matrix

    acc = accuracy_score(test_label, y_pred)
    print("Acc:", acc)
    rec = recall_score(test_label, y_pred)
    print("Rec:", rec)
    pre = precision_score(test_label, y_pred)
    print("Pre:", pre)
    f1s = f1_score(test_label, y_pred)
    print("F1s:", f1s)
    cm = confusion_matrix(test_label, y_pred)

    tn, fp, fn, tp = confusion_matrix(test_label, y_pred).ravel()

    # Fall out or false positive rate
    FPR = fp/(fp+tn)
    # False negative rate
    FNR = fn/(tp+fn)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    print("FPR:", FPR)
    print("FNR:", FNR)

    y_pred = clf.predict_threshold(test_data, threshold=THRESHOLD)

    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    from sklearn.metrics import confusion_matrix

    acc = accuracy_score(test_label, y_pred)
    print("Acc:", acc)
    rec = recall_score(test_label, y_pred)
    print("Rec:", rec)
    pre = precision_score(test_label, y_pred)
    print("Pre:", pre)
    f1s = f1_score(test_label, y_pred)
    print("F1s:", f1s)
    cm = confusion_matrix(test_label, y_pred)

    tn, fp, fn, tp = confusion_matrix(test_label, y_pred).ravel()

    # Fall out or false positive rate
    FPR = fp/(fp+tn)
    # False negative rate
    FNR = fn/(tp+fn)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    print("FPR:", FPR)
    print("FNR:", FNR)

    adv_attributes = []
    # walk in test features
    for input in adv_files:
        
        print("Reading {}...".format(input))

        # read input file
        file = open(input, 'r') 
        # read its lines
        sws = file.readlines() 
        
        # walk in each sw
        for sw in sws: 
            # initialize extractor
            # at_extractor = JSONAttributeExtractor(sw)
            # # get adv_attributes
            # atts = at_extractor.extract()
            atts = json.loads(sw)
            # save attribute
            adv_attributes.append(atts)

        # close file
        file.close()

    adv_data = pd.DataFrame(adv_attributes)
    adv_data = adv_data[(adv_data["label"]==1) | (adv_data["label"]==0)]
    #print(adv_data)
    print(adv_data.shape)

    adv_label = adv_data["label"].values
    y_pred = clf.predict(adv_data)

    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    from sklearn.metrics import confusion_matrix

    acc = accuracy_score(adv_label, y_pred)
    print("Acc:", acc)
    rec = recall_score(adv_label, y_pred)
    print("Rec:", rec)
    pre = precision_score(adv_label, y_pred)
    print("Pre:", pre)
    f1s = f1_score(adv_label, y_pred)
    print("F1s:", f1s)
    cm = confusion_matrix(adv_label, y_pred)

    tn, fp, fn, tp = confusion_matrix(adv_label, y_pred).ravel()

    # Fall out or false positive rate
    FPR = fp/(fp+tn)
    # False negative rate
    FNR = fn/(tp+fn)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    print("FPR:", FPR)
    print("FNR:", FNR)
    y_pred = clf.predict_threshold(adv_data, threshold=THRESHOLD)

    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    from sklearn.metrics import confusion_matrix

    acc = accuracy_score(adv_label, y_pred)
    print("Acc:", acc)
    rec = recall_score(adv_label, y_pred)
    print("Rec:", rec)
    pre = precision_score(adv_label, y_pred)
    print("Pre:", pre)
    f1s = f1_score(adv_label, y_pred)
    print("F1s:", f1s)
    cm = confusion_matrix(adv_label, y_pred)

    tn, fp, fn, tp = confusion_matrix(adv_label, y_pred).ravel()

    # Fall out or false positive rate
    FPR = fp/(fp+tn)
    # False negative rate
    FNR = fn/(tp+fn)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    print("FPR:", FPR)
    print("FNR:", FNR)
