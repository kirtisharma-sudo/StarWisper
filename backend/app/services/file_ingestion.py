import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import re
from ..models.schemas import LightCurveMetadata, ProcessedData
from ..utils.logging import get_logger

logger = get_logger(__name__)

class FileIngestionService:
    @staticmethod
    def extract_metadata(file_path: str) -> LightCurveMetadata:
        ext = Path(file_path).suffix.lower()
        if ext in [".fits", ".fit"]:
            return FileIngestionService._extract_fits_metadata(file_path)
        elif ext in [".csv", ".txt"]:
            return FileIngestionService._extract_csv_metadata(file_path)
        else:
            raise ValueError("Unsupported file type")

    @staticmethod
    def _extract_fits_metadata(file_path: str) -> LightCurveMetadata:
        with fits.open(file_path) as hdul:
            if len(hdul) > 1:
                data = hdul[1].data
                header = hdul[1].header
            else:
                data = hdul[0].data
                header = hdul[0].header
            columns = data.dtype.names
            time_col = None
            for col in columns:
                if re.match(r'TIME|time|Time', col):
                    time_col = col
                    break
            if time_col is None:
                if len(columns) > 0 and 'time' in columns[0].lower():
                    time_col = columns[0]
                else:
                    raise ValueError("No TIME column found")
            flux_col = None
            for col in columns:
                if re.match(r'(PDCSAP|SAP)_FLUX|flux|Flux', col):
                    flux_col = col
                    break
            if flux_col is None:
                idx = list(columns).index(time_col)
                if len(columns) > idx+1:
                    flux_col = columns[idx+1]
                else:
                    raise ValueError("No FLUX column found")
            quality_col = None
            for col in columns:
                if re.match(r'QUALITY|quality', col):
                    quality_col = col
                    break
            time = data[time_col]
            flux = data[flux_col]
            if quality_col is not None:
                quality = data[quality_col]
            else:
                quality = None
            return LightCurveMetadata(
                target=header.get('OBJECT', header.get('TICID', None)),
                ra=header.get('RA_OBJ', None),
                dec=header.get('DEC_OBJ', None),
                tic=header.get('TICID', None),
                sector=header.get('SECTOR', None),
                camera=header.get('CAMERA', None),
                ccd=header.get('CCD', None),
                time_columns=[time_col],
                flux_columns=[flux_col],
                quality_column=quality_col,
                n_points=len(time),
                time_min=float(np.nanmin(time)),
                time_max=float(np.nanmax(time))
            )

    @staticmethod
    def _extract_csv_metadata(file_path: str) -> LightCurveMetadata:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        time_col = None
        for col in columns:
            if re.match(r'TIME|time|Time', col):
                time_col = col
                break
        if time_col is None:
            time_col = columns[0]
        flux_col = None
        for col in columns:
            if re.match(r'(PDCSAP|SAP)_FLUX|flux|Flux', col):
                flux_col = col
                break
        if flux_col is None:
            idx = columns.index(time_col)
            if len(columns) > idx+1:
                flux_col = columns[idx+1]
            else:
                raise ValueError("No FLUX column found")
        quality_col = None
        for col in columns:
            if re.match(r'QUALITY|quality', col):
                quality_col = col
                break
        time = df[time_col].values
        flux = df[flux_col].values
        return LightCurveMetadata(
            target=None,
            time_columns=[time_col],
            flux_columns=[flux_col],
            quality_column=quality_col,
            n_points=len(time),
            time_min=float(np.nanmin(time)),
            time_max=float(np.nanmax(time))
        )

    @staticmethod
    def load_processed_result(result_path: str) -> ProcessedData:
        data = np.load(result_path, allow_pickle=True)
        metadata_dict = data['metadata'].item() if 'metadata' in data else {}
        return ProcessedData(
            time=data['time'].tolist(),
            flux=data['flux'].tolist(),
            flux_err=data.get('flux_err', None).tolist() if 'flux_err' in data else None,
            detrended_flux=data.get('detrended_flux', None).tolist() if 'detrended_flux' in data else None,
            quality_mask=data.get('quality_mask', None).tolist() if 'quality_mask' in data else None,
            metadata=LightCurveMetadata(**metadata_dict)
        )
