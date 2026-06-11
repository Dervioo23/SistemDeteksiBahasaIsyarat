import os
import numpy as np
import pickle
from typing import Dict, List, Tuple

from preprocessing import (
    load_multiple_gestures,
    print_dataset_summary,
    normalize_batch,
    pad_sequences_to_max_length,
    augment_dataset,
    stratified_split,
    print_split_info,
    enhanced_split,
    print_enhanced_split_info,
    analyze_participant_distribution
)


class DataPreprocessor:
    """Pipeline prapemrosesan data lengkap"""
    
    def __init__(self, config: Dict = None):
        """
        Inisialisasi preprocessor
        
        Args:
            config: Dictionary konfigurasi
        """
        self.config = config or self.get_default_config()
        self.class_names = []
        self.num_classes = 0
        
    def get_default_config(self) -> Dict:
        """Dapatkan konfigurasi prapemrosesan default"""
        return {
            'data_dir': 'dataset',
            'output_dir': 'preprocessed_data',
            'normalize': True,
            'normalization_method': 'full',
            'augment': True,
            'augmentation_factor': 2,
            'pad_sequences': True,
            'max_length': 45,  # Untuk urutan kata
            'train_ratio': 0.67,
            'val_ratio': 0.17,
            'test_ratio': 0.16,
            'random_seed': 42,
            'split_method': 'auto',  # 'auto', 'participant', 'session', 'stratified'
            'prevent_data_leakage': True,  # Aktifkan pemisahan yang ditingkatkan
            # Penugasan peserta manual (opsional)
            'train_participants': None,  # BARU: ['001', '002'] untuk penugasan manual
            'val_participants': None,    # BARU: ['003'] untuk penugasan manual
            'test_participants': None,   # BARU: ['004'] untuk penugasan manual
            'analyze_participants': True # BARU: Tampilkan analisis peserta sebelum pemisahan
        }
    
    def load_data(
        self,
        labels: List[str],
        category: str = 'words'
    ) -> Tuple[List[np.ndarray], List[int], List[str], List[Dict]]:
        """
        Muat data dari dataset
        
        Args:
            labels: Daftar label gesture untuk dimuat
            category: 'words' atau 'alphabet'
            
        Returns:
            Tuple dari (samples, labels, file_ids, metadatas)
        """
        print(f"\n{'='*60}")
        print(f"LOADING DATA - {category.upper()}")
        print(f"{'='*60}")
        
        samples, class_indices, file_ids, metadatas = load_multiple_gestures(
            data_dir=self.config['data_dir'],
            labels=labels,
            category=category
        )
        
        self.class_names = labels
        self.num_classes = len(labels)
        
        return samples, class_indices, file_ids, metadatas
    
    def preprocess_data(
        self,
        samples: List[np.ndarray],
        labels: List[int],
        normalize: bool = None,
        augment: bool = None,
        pad: bool = None
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Terapkan pipeline prapemrosesan
        
        Args:
            samples: Daftar array landmark
            labels: Daftar label kelas
            normalize: Apakah akan menormalisasi (timpa config)
            augment: Apakah akan mengaugmentasi (timpa config)
            pad: Apakah akan mem-pad urutan (timpa config)
            
        Returns:
            Tuple dari (preprocessed_samples, labels)
        """
        if normalize is None:
            normalize = self.config['normalize']
        if augment is None:
            augment = self.config['augment']
        if pad is None:
            pad = self.config['pad_sequences']
        
        print(f"\n{'='*60}")
        print("PREPROCESSING PIPELINE")
        print(f"{'='*60}")
        
        # Langkah 1: Normalisasi
        if normalize:
            method = self.config.get('normalization_method', 'full')
            print("\n📍 Step 1: Normalizing landmarks (method: {0})...".format(method))
            samples = normalize_batch(samples, method=method)
            print("   ✅ Normalization complete")
        
        return samples, labels
    
    def split_data(
        self,
        samples: List[np.ndarray],
        labels: List[int],
        file_ids: List[str] = None
    ) -> Dict:
        """
        Bagi data menjadi set train/val/test dengan metode yang ditingkatkan
        
        Args:
            samples: Daftar array landmark
            labels: Daftar label kelas
            file_ids: Daftar ID file untuk pemisahan berbasis peserta/sesi
            
        Returns:
            Dictionary dengan pemisahan train/val/test dan split_info
        """
        print(f"\n{'='*60}")
        print("ENHANCED DATA SPLITTING")
        print(f"{'='*60}")
        
        # Gunakan pemisahan yang ditingkatkan jika diaktifkan dan file_ids tersedia
        if self.config.get('prevent_data_leakage', False) and file_ids:
            
            # Tampilkan analisis peserta jika diminta
            if self.config.get('analyze_participants', True):
                analyze_participant_distribution(
                    file_ids=file_ids,
                    labels=labels,
                    class_names=self.class_names
                )
            
            # Siapkan argumen pemisahan yang ditingkatkan
            split_kwargs = {
                'samples': samples,
                'labels': labels,
                'file_ids': file_ids,
                'split_method': self.config.get('split_method', 'auto'),
                'train_ratio': self.config['train_ratio'],
                'val_ratio': self.config['val_ratio'],
                'test_ratio': self.config['test_ratio'],
                'random_seed': self.config['random_seed']
            }
            
            # Tambahkan penugasan peserta manual jika disediakan
            if self.config.get('train_participants') or self.config.get('val_participants') or self.config.get('test_participants'):
                print(f"\n🎯 Using MANUAL PARTICIPANT ASSIGNMENT from config")
                split_kwargs['train_participants'] = self.config.get('train_participants')
                split_kwargs['val_participants'] = self.config.get('val_participants') 
                split_kwargs['test_participants'] = self.config.get('test_participants')
                split_kwargs['auto_split_participants'] = False
            
            (train_X, train_y, train_file_ids,
             val_X, val_y, val_file_ids,
             test_X, test_y, test_file_ids,
             split_info) = enhanced_split(**split_kwargs)
            
            print_enhanced_split_info(
                train_y, val_y, test_y,
                train_file_ids, val_file_ids, test_file_ids,
                split_info, class_names=self.class_names
            )
            
            return {
                'train': (train_X, train_y),
                'val': (val_X, val_y),
                'test': (test_X, test_y),
                'split_info': split_info,
                'file_ids': {
                    'train': train_file_ids,
                    'val': val_file_ids,
                    'test': test_file_ids
                }
            }
        
        else:
            # Fallback ke pemisahan bertingkat tradisional
            print("⚠️  Using traditional stratified split (file_ids not available or enhanced splitting disabled)")
            
            train_X, train_y, val_X, val_y, test_X, test_y = stratified_split(
                samples=samples,
                labels=labels,
                train_ratio=self.config['train_ratio'],
                val_ratio=self.config['val_ratio'],
                test_ratio=self.config['test_ratio'],
                random_seed=self.config['random_seed']
            )
            
            print_split_info(train_y, val_y, test_y, class_names=self.class_names)
            
            return {
                'train': (train_X, train_y),
                'val': (val_X, val_y),
                'test': (test_X, test_y),
                'split_info': {'split_method': 'stratified_fallback'}
            }
    
    def convert_to_arrays(
        self,
        samples: List[np.ndarray],
        labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Konversi daftar ke array numpy
        
        Args:
            samples: Daftar array landmark
            labels: Daftar label kelas
            
        Returns:
            Tuple dari (X, y) sebagai array numpy
        """
        # Tumpuk sampel
        X = np.array(samples, dtype=np.float32)
        
        # Konversi label ke array
        y = np.array(labels, dtype=np.int32)
        
        return X, y
    
    def save_preprocessed_data(
        self,
        data_splits: Dict,
        output_dir: str = None
    ):
        """
        Simpan data yang telah diproses ke disk
        
        Args:
            data_splits: Dictionary dengan pemisahan train/val/test
            output_dir: Direktori output (timpa config)
        """
        if output_dir is None:
            output_dir = self.config['output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("SAVING PREPROCESSED DATA")
        print(f"{'='*60}")
        
        # Simpan setiap pemisahan (filter entri metadata)
        valid_splits = ['train', 'val', 'test']
        
        for split_name in valid_splits:
            if split_name in data_splits:
                samples, labels = data_splits[split_name]
                
                # Konversi ke array
                X, y = self.convert_to_arrays(samples, labels)
                
                # Simpan
                X_path = os.path.join(output_dir, f'{split_name}_X.npy')
                y_path = os.path.join(output_dir, f'{split_name}_y.npy')
                
                np.save(X_path, X)
                np.save(y_path, y)
                
                print(f"\n✅ {split_name.upper()} set saved:")
                print(f"   X: {X_path} (shape: {X.shape})")
                print(f"   y: {y_path} (shape: {y.shape})")
        
        # Simpan metadata (dapatkan info sampel dari set train)
        train_samples = data_splits.get('train', ([], []))[0]
        
        # Simpan metadata yang ditingkatkan termasuk info pemisahan
        metadata = {
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'num_features': train_samples[0].shape[-1] if train_samples else 0,
            'max_length': self.config['max_length'],
            'config': self.config,
            'split_info': data_splits.get('split_info', {}),
            'split_method': data_splits.get('split_info', {}).get('split_method', 'unknown')
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✅ Metadata saved: {metadata_path}")
        
        # Simpan file_ids jika tersedia (untuk pemisahan yang ditingkatkan)
        if 'file_ids' in data_splits:
            file_ids_path = os.path.join(output_dir, 'file_ids.pkl')
            with open(file_ids_path, 'wb') as f:
                pickle.dump(data_splits['file_ids'], f)
            print(f"✅ File IDs saved: {file_ids_path}")
            
            # Juga simpan info pemisahan sebagai file terpisah untuk akses mudah
            if 'split_info' in data_splits:
                split_info_path = os.path.join(output_dir, 'split_info.pkl')
                with open(split_info_path, 'wb') as f:
                    pickle.dump(data_splits['split_info'], f)
                print(f"✅ Split info saved: {split_info_path}")
        
        print(f"{'='*60}\n")
    
    def run_full_pipeline(
        self,
        labels: List[str],
        category: str = 'words'
    ) -> Dict:
        """
        Jalankan pipeline prapemrosesan lengkap
        
        Args:
            labels: Daftar label gesture
            category: 'words' atau 'alphabet'
            
        Returns:
            Dictionary dengan pemisahan data yang telah diproses
        """
        # Langkah 1: Muat data
        samples, class_indices, file_ids, metadatas = self.load_data(
            labels=labels,
            category=category
        )
        
        if not samples:
            print("❌ No data loaded! Check dataset directory.")
            return {}
        
        # Langkah 2: Prapemrosesan
        samples, class_indices = self.preprocess_data(
            samples=samples,
            labels=class_indices
        )
        
        # Langkah 3: Pemisahan (dengan file_ids untuk enhanced splitting)
        data_splits = self.split_data(
            samples=samples,
            labels=class_indices,
            file_ids=file_ids
        )

        # Langkah 4: Augmentasi HANYA TRAIN (pasca-pemisahan)
        if self.config.get('augment', True):
            train_X, train_y = data_splits['train']
            augmentation_config = {
                'rotation': True,
                'scale': True,
                'translation': True,
                'noise': True,
                'temporal_stretch': True,
                'flip': False,
                'hand_dropout': True,
                'hand_dropout_prob': 0.5,
            }
            train_X_aug, train_y_aug = augment_dataset(
                samples=train_X,
                labels=train_y,
                augmentation_factor=self.config['augmentation_factor'],
                augmentation_config=augmentation_config
            )
            data_splits['train'] = (train_X_aug, train_y_aug)

        # Langkah 5: Pad urutan per pemisahan (jika diperlukan)
        def pad_if_needed(split_samples: List[np.ndarray]) -> List[np.ndarray]:
            if not split_samples:
                return split_samples
            if any(s.shape[0] != split_samples[0].shape[0] for s in split_samples):
                samples_array = pad_sequences_to_max_length(
                    sequences=split_samples,
                    max_length=self.config['max_length'],
                    padding='post'
                )
                return [samples_array[i] for i in range(len(samples_array))]
            return split_samples

        data_splits['train'] = (pad_if_needed(data_splits['train'][0]), data_splits['train'][1])
        data_splits['val'] = (pad_if_needed(data_splits['val'][0]), data_splits['val'][1])
        data_splits['test'] = (pad_if_needed(data_splits['test'][0]), data_splits['test'][1])

        # Langkah 6: Simpan
        self.save_preprocessed_data(data_splits)
        
        print("\n🎉 PREPROCESSING COMPLETE!")
        print(f"   Output directory: {self.config['output_dir']}")
        print(f"   Ready for training!\n")
        
        return data_splits


def main():
    """Fungsi utama untuk prapemrosesan"""
    
    print("\n" + "="*60)
    print("SIGN LANGUAGE DETECTION - DATA PREPROCESSING")
    print("="*60)
    
    # Konfigurasi
    config = {
        'data_dir': 'dataset',
        'output_dir': 'preprocessed_data',
        'normalize': True,
        'normalization_method': 'full',
        'augment': True,
        'augmentation_factor': 2,  # Augmentasi 2x
        'pad_sequences': True,
        'max_length': 45,
        'train_ratio': 0.67,
        'val_ratio': 0.17,
        'test_ratio': 0.16,
        'random_seed': 42
    }
    
    # Inisialisasi preprocessor
    preprocessor = DataPreprocessor(config=config)
    
    # Tampilkan ringkasan dataset
    from preprocessing import print_dataset_summary
    print_dataset_summary(config['data_dir'])
    
    # Tentukan gesture untuk diproses
    print("\n" + "="*60)
    print("SELECT GESTURES TO PREPROCESS")
    print("="*60)
    print("\n1. Words: halo")
    print("2. Alphabet: C")
    print("3. Both (combined dataset)")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        # Proses kata saja
        labels = ['halo']
        category = 'words'
        preprocessor.run_full_pipeline(labels=labels, category=category)
    
    elif choice == '2':
        # Proses alfabet saja
        labels = ['C']
        category = 'alphabet'
        preprocessor.run_full_pipeline(labels=labels, category=category)
    
    elif choice == '3':
        # Proses keduanya - perlu penanganan terpisah karena bentuk berbeda
        print("\n⚠️  Note: Word (45 frames) and Alphabet (1 frame) have different shapes.")
        print("   They will be preprocessed separately.\n")
        
        # Proses kata
        print("\n" + "="*60)
        print("PROCESSING WORDS")
        print("="*60)
        config['output_dir'] = 'preprocessed_data/words'
        preprocessor_words = DataPreprocessor(config=config)
        preprocessor_words.run_full_pipeline(labels=['halo'], category='words')
        
        # Proses alfabet
        print("\n" + "="*60)
        print("PROCESSING ALPHABET")
        print("="*60)
        config['output_dir'] = 'preprocessed_data/alphabet'
        config['max_length'] = 1  # Alfabet statis
        preprocessor_alphabet = DataPreprocessor(config=config)
        preprocessor_alphabet.run_full_pipeline(labels=['C'], category='alphabet')
    
    else:
        print("Invalid choice!")


if __name__ == '__main__':
    main()
