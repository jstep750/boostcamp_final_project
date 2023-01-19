import pandas as pd
import numpy as np
import konlpy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

from datasketch import MinHash

# tmp
import pickle

class NewsCluster:

    def __init__(self, min_df = 1, ngram_range=(1,1)):
        self.okt = konlpy.tag.Okt()
        self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)

    def cluster(self,
                data,
                num_clusters):

        documents_processed = []

        for each in data:

            text = each['_source']['titleNdescription']
            okt_pos = self.okt.pos(text)
            
            words = []
            for word, pos in okt_pos:
                if 'Noun' in pos:
                    words.append(word)

            documents_processed.append(' '.join(words))

        tfidf_vector = self.tfidf_vectorizer.fit_transform(documents_processed)
        tfidf_dense = tfidf_vector.todense()

        # type matrix to type np.array
        tfidf_dense = np.squeeze(np.asarray(tfidf_dense))

        model = AgglomerativeClustering(linkage='complete', affinity='cosine', n_clusters=num_clusters)
        model.fit(tfidf_dense)

        for each_data, label in zip(data, model.labels_.tolist()):
            each_data['_source']['label'] = label

        return data

class RemoveDup:

    # 중복 기사도 제거 하는 방법이 다양

    def __init__(self):

        self.m1 = MinHash()
        self.m2 = MinHash()

    def remove_dup(self,
                    title_N_desc,
                    threshold=0.1):

        total_Jaccard = []
        for i in range(len(title_N_desc)):

            Jaccard = []
            data1 = title_N_desc[i].split()
            s1 = set(data1)

            for j in range(len(title_N_desc)):
                if i == j:
                    Jaccard.append(1)
                    continue
                data2 = title_N_desc[j].split()

                ## Estimated Jaccard for data1 and dat2
                # for each in data1:
                #     m1.update(each.encode('utf8'))
                
                # for eachin data2:
                #     m2.update(each.encode('utf8'))

                # Jaccard.append(m1.jaccard(m2))

                s2 = set(data2)

                actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
                Jaccard.append(actual_jaccard)

            total_Jaccard.append(Jaccard)

        remove_list = []
        for i in range(len(total_Jaccard)):
            for j in range(i+1, len(total_Jaccard)):
                if total_Jaccard[i][j] > threshold:
                    remove_list.append(j)

        return remove_list

if __name__ == '__main__':


    removedup = RemoveDup()
    removedup.remove_dup()

