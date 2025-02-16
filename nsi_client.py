import aiohttp
import asyncio
import pandas as pd
import urllib3
import zipfile
import io
import json
from typing import Union, List, Dict, Optional
from dataclasses import dataclass
import nest_asyncio

# Применяем nest_asyncio для работы с уже запущенным event loop (например, в Jupyter)
nest_asyncio.apply()

@dataclass
class MetadataResult:
    oid: str
    metadata: Optional[Dict]
    error: Optional[str] = None

class NSIClient:
    def __init__(self, token: str):
        self.metadata_cache: Dict[str, Dict] = {}
        self.base_url: str = "https://nsi.rosminzdrav.ru/port/rest"
        self.download_url: str = "https://nsi.rosminzdrav.ru/api/dataFiles"
        self.token: str = token
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.oid_dictionary: Dict[str, str] = {}

    async def get_minimal_metadata(self, session: aiohttp.ClientSession, oid: str) -> MetadataResult:
        """
        Асинхронное получение метаданных для одного OID.
        Используем параметры сортировки по версии (desc), чтобы выбрать последнюю актуальную версию.
        Если API возвращает список – выбирается первый элемент.
        """
        try:
            params = {
                'userKey': self.token,
                'identifier': oid,
                'sort': 'version',
                'direction': 'desc'
            }
            async with session.get(f"{self.base_url}/passport", params=params, ssl=False) as response:
                if response.status != 200:
                    return MetadataResult(oid, None, f"Ошибка запроса: {response.status}")
                data = await response.json()
                # Если API возвращает список версий, выбираем первый элемент
                if isinstance(data, list) and data:
                    latest = data[0]
                else:
                    latest = data
                metadata = {
                    'shortName': latest.get('shortName', 'Неизвестный справочник'),
                    'version': latest.get('version')
                }
                # Добавляем в словарь соответствия OID и shortName
                self.oid_dictionary[oid] = metadata['shortName']
                return MetadataResult(oid, metadata)
        except Exception as e:
            return MetadataResult(oid, None, str(e))

    async def get_all_metadata(self, oids: List[str]) -> Dict[str, Dict]:
        """
        Асинхронное получение всех метаданных для списка OID с использованием кэша.
        Если для какого-либо OID метаданные ещё не получены, происходит запрос через API.
        """
        uncached_oids = [oid for oid in oids if oid not in self.metadata_cache]
        if uncached_oids:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                tasks = [self.get_minimal_metadata(session, oid) for oid in uncached_oids]
                results = await asyncio.gather(*tasks)
                for result in results:
                    if not result.error and result.metadata is not None:
                        self.metadata_cache[result.oid] = result.metadata
                    else:
                        print(f"Ошибка получения метаданных для {result.oid}: {result.error}")
        
        # Сохраняем словарь OID после каждого обновления метаданных
        self.save_oid_dictionary()
        return {oid: self.metadata_cache.get(oid) for oid in oids if oid in self.metadata_cache}

    def save_oid_dictionary(self, filename: str = "oid_dictionary.json"):
        """
        Сохраняет словарь соответствия OID и shortName в JSON файл
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.oid_dictionary, f, ensure_ascii=False, indent=2)
        print(f"Словарь OID сохранен в файл: {filename}")

    async def download_csv(self, session: aiohttp.ClientSession, oid: str, version: str) -> pd.DataFrame:
        """
        Асинхронная загрузка и распаковка CSV-файла.
        URL формируется по шаблону: /api/dataFiles/{oid}_{version}_csv.zip.
        """
        url = f"{self.download_url}/{oid}_{version}_csv.zip"
        print(f"Скачивание файла: {url}")
        async with session.get(url, ssl=False) as response:
            if response.status != 200:
                raise Exception(f"Ошибка загрузки файла: {response.status}")
            content = await response.read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as csv_file:
                    try:
                        return pd.read_csv(csv_file, sep=';')
                    except Exception:
                        csv_file.seek(0)
                        return pd.read_csv(csv_file, sep=',')

    async def process_oids(self, oids: List[str]) -> None:
        """
        Асинхронная обработка списка OID.
        Для каждого OID из списка сначала получаются метаданные (и выбирается актуальная версия),
        затем скачивается CSV, и файл сохраняется с именем, полученным из OID (точки заменены на _).
        """
        metadata_dict = await self.get_all_metadata(oids)
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for oid in oids:
                if oid not in metadata_dict:
                    print(f"Метаданные для {oid} не получены.")
                    continue
                metadata = metadata_dict[oid]
                print(f"\n=== Справочник: {metadata['shortName']} ===")
                try:
                    df = await self.download_csv(session, oid, metadata['version'])
                    filename = oid.replace('.', '_') + ".csv"
                    df.to_csv(filename, index=False)
                    print(f"Файл сохранен: {filename}")
                except Exception as e:
                    print(f"Ошибка обработки OID {oid}: {str(e)}")

    def process_oid(self, oid: Union[str, List[str]], to_dataframe: bool = False) -> Union[pd.DataFrame, bool, None]:
        """
        Синхронный метод-обёртка для обработки одного или нескольких OID.
        Если указан одиночный OID и параметр to_dataframe=True, возвращается DataFrame.
        Для списка OID данные сохраняются в файлы.
        """
        return asyncio.run(self.async_process_oid(oid, to_dataframe))

    async def async_process_oid(self, oid: Union[str, List[str]], to_dataframe: bool = False) -> Union[pd.DataFrame, bool, None]:
        """
        Асинхронная обработка одного или нескольких OID.
        """
        if isinstance(oid, list):
            await self.process_oids(oid)
            return True
        else:
            metadata_dict = await self.get_all_metadata([oid])
            if oid in metadata_dict:
                metadata = metadata_dict[oid]
                print(f"\n=== Справочник: {metadata['shortName']} ===")
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    try:
                        df = await self.download_csv(session, oid, metadata['version'])
                        if to_dataframe:
                            return df
                        else:
                            filename = oid.replace('.', '_') + ".csv"
                            df.to_csv(filename, index=False)
                            print(f"Файл сохранен: {filename}")
                            return True
                    except Exception as e:
                        print(f"Ошибка обработки OID {oid}: {str(e)}")
                        return None
            return None

def main():
    """
    Асинхронный запуск программы в интерактивном режиме.
    Пользователь вводит один или несколько OID через запятую.
    """
    token = "b06384c3-e4f8-497f-a38e-651ec01c99a7"
    client = NSIClient(token)
    while True:
        oid_input = input("Введите OID или несколько OID через запятую (или 'exit' для выхода): ").strip()
        if oid_input.lower() == 'exit':
            break
        oids = [oid.strip() for oid in oid_input.split(',')]
        asyncio.run(client.process_oids(oids))

if __name__ == '__main__':
    main()
