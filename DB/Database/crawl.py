import os
import gdown
import zipfile
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

class CrawlNews:

    def __init__(self):
        self.dataset_folder = 'Financial_news'

    def crawl_from_file(self):

        base_url = 'https://drive.google.com/uc?id='
        data_folder_id = '1KwAa1Huh7b4D3Mp6L0OFSjUrxum7b1an' # Shared dataset folder id in my google drive
        url = ''.join([base_url, data_folder_id])
        output_name = 'Newsdataset.zip'

        gdown.download(url, output_name, quiet=False)

        tmp = zipfile.ZipFile('Newsdataset.zip')
        tmp.extractall('Financial_news')
        tmp.close()

        # remove zip file
        os.remove('Newsdataset.zip')

    def crawl_from_seleninum(self):
        
        # TODO
        # 필요하면 만들 예정

        pass

    def get_news_dataframe(self):

        file_list = os.listdir('Financial_news')

        total_df = pd.DataFrame()

        for each in tqdm(file_list, desc='concatenating news dataframes'):
            tmp_df = pd.read_excel(f'Financial_news/{each}')
            total_df = pd.concat([total_df, tmp_df])

        breakpoint()
        total_df = total_df.loc[~total_df['URL'].isna()]

        def url_check(x):

            http_str = ('http://', 'https://')

            if not x.startswith(http_str):
                print("not startwith https")
                return 'http://'+x
            else:
                return x

        total_df['URL'] = total_df['URL'].apply(url_check)
        total_df = total_df.reset_index(drop=True)

        return total_df

if __name__ == '__main__':

    cw = CrawlNews()
    cw.get_news_dataframe()