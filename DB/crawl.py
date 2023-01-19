import os
import gdown
import zipfile

class CrawlNews:

    def __init__(self):
        self.dataset_folder = 'Financial_news'

    def crawl_from_file(self):

        base_url = 'https://drive.google.com/uc?id='
        data_folder_id = '1nUBCCPH2n6JvWYZotdYFy-z9JKra5N8k' # Shared dataset folder id in my google drive
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

    # 그 문자들로 해시를 해서 넣을지 말지를 결정을 해야하는데

if __name__ == '__main__':

    cw = CrawlNews()
    cw.crawl_from_file()