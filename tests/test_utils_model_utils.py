"""
Unit tests for utils/model_utils.py

Tests model loading and device management utilities.
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.utils.model_utils import (
    get_device,
    load_model_with_device,
    load_voice_clone_model,
    load_custom_voice_model,
    load_voice_design_model,
    get_model_memory_usage,
    clear_gpu_cache,
    get_device_info,
    validate_model_path
)


class TestModelUtilities(unittest.TestCase):
    """Test model loading and management utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_model_path = os.path.join(self.temp_dir, "test_model")
        os.makedirs(self.test_model_path)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    @patch('torch.device')
    def test_get_device_cuda_available(self, mock_device, mock_get_name, mock_cuda_available):
        """Test get_device when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_get_name.return_value = "Test GPU"
        mock_device.return_value = "cuda:0"
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = get_device()
        
        mock_device.assert_called_with("cuda")
        mock_get_name.assert_called_once_with(0)
        self.assertEqual(result, "cuda:0")
    
    @patch('torch.cuda.is_available')
    @patch('torch.device')
    def test_get_device_cuda_unavailable(self, mock_device, mock_cuda_available):
        """Test get_device when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        mock_device.return_value = "cpu"
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = get_device()
        
        mock_device.assert_called_with("cpu")
        self.assertEqual(result, "cpu")
    
    @patch('src.utils.model_utils.Qwen3TTSModel')
    @patch('src.utils.model_utils.get_device')
    @patch('src.utils.model_utils.tqdm', None)  # Test without tqdm
    def test_load_model_with_device_no_tqdm(self, mock_get_device, mock_model_class):
        """Test load_model_with_device without tqdm."""
        mock_device = MagicMock()
        mock_get_device.return_value = mock_device
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = load_model_with_device(self.test_model_path, "TestModel")
        
        mock_model_class.from_pretrained.assert_called_once_with(
            self.test_model_path,
            device_map={"": mock_device},
            dtype=unittest.mock.ANY
        )
        self.assertEqual(result, mock_model)
    
    @patch('src.utils.model_utils.Qwen3TTSModel')
    @patch('src.utils.model_utils.get_device')
    @patch('src.utils.model_utils.tqdm')
    def test_load_model_with_device_with_tqdm(self, mock_tqdm_module, mock_get_device, mock_model_class):
        """Test load_model_with_device with tqdm."""
        mock_device = MagicMock()
        mock_get_device.return_value = mock_device
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock tqdm context manager
        mock_tqdm = MagicMock()
        mock_tqdm_module.return_value.__enter__.return_value = mock_tqdm
        mock_tqdm_module.return_value.__exit__.return_value = None
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = load_model_with_device(self.test_model_path, "TestModel")
        
        mock_tqdm_module.assert_called_once()
        mock_tqdm.update.assert_called_once_with(1)
        self.assertEqual(result, mock_model)
    
    @patch('src.utils.model_utils.Qwen3TTSModel', None)  # Simulate missing import
    def test_load_model_with_device_no_qwen_tts(self):
        """Test load_model_with_device when Qwen3TTSModel is unavailable."""
        with self.assertRaises(ImportError) as context:
            load_model_with_device(self.test_model_path)
        
        self.assertIn("qwen_tts not installed", str(context.exception))
    
    @patch('src.utils.model_utils.Qwen3TTSModel')
    @patch('src.utils.model_utils.get_device')
    def test_load_model_with_device_loading_error(self, mock_get_device, mock_model_class):
        """Test load_model_with_device with model loading error."""
        mock_get_device.return_value = MagicMock()
        mock_model_class.from_pretrained.side_effect = Exception("Loading failed")
        
        with patch('sys.stdout'):  # Suppress progress messages
            with self.assertRaises(Exception) as context:
                load_model_with_device(self.test_model_path)
        
        self.assertIn("Loading failed", str(context.exception))
    
    @patch('src.utils.model_utils.load_model_with_device')
    def test_load_voice_clone_model(self, mock_load):
        """Test load_voice_clone_model calls load_model_with_device correctly."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        result = load_voice_clone_model("test_path")
        
        mock_load.assert_called_once_with("test_path", model_name="Voice Clone")
        self.assertEqual(result, mock_model)
    
    @patch('src.utils.model_utils.load_model_with_device')
    def test_load_custom_voice_model(self, mock_load):
        """Test load_custom_voice_model calls load_model_with_device correctly."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        result = load_custom_voice_model("test_path")
        
        mock_load.assert_called_once_with("test_path", model_name="CustomVoice")
        self.assertEqual(result, mock_model)
    
    @patch('src.utils.model_utils.load_model_with_device')
    def test_load_voice_design_model(self, mock_load):
        """Test load_voice_design_model calls load_model_with_device correctly."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        result = load_voice_design_model("test_path")
        
        mock_load.assert_called_once_with("test_path", model_name="VoiceDesign")
        self.assertEqual(result, mock_model)
    
    @patch('torch.cuda.is_available')
    def test_get_model_memory_usage_no_cuda(self, mock_cuda_available):
        """Test get_model_memory_usage when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        
        result = get_model_memory_usage()
        
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["memory_info"], "N/A (using CPU)")
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.get_device_name')
    def test_get_model_memory_usage_cuda(self, mock_get_name, mock_get_props, 
                                        mock_memory_reserved, mock_memory_allocated,
                                        mock_current_device, mock_cuda_available):
        """Test get_model_memory_usage when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_current_device.return_value = 0
        mock_memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_memory_reserved.return_value = 2 * 1024 * 1024 * 1024  # 2GB
        mock_get_name.return_value = "Test GPU"
        
        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_get_props.return_value = mock_props
        
        result = get_model_memory_usage()
        
        self.assertEqual(result["device"], "cuda:0")
        self.assertEqual(result["device_name"], "Test GPU")
        self.assertEqual(result["memory_allocated_gb"], 1.0)
        self.assertEqual(result["memory_reserved_gb"], 2.0)
        self.assertEqual(result["memory_total_gb"], 8.0)
        self.assertEqual(result["memory_free_gb"], 6.0)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_clear_gpu_cache_cuda_available(self, mock_empty_cache, mock_cuda_available):
        """Test clear_gpu_cache when CUDA is available."""
        mock_cuda_available.return_value = True
        
        with patch('sys.stdout'):  # Suppress progress messages
            clear_gpu_cache()
        
        mock_empty_cache.assert_called_once()
    
    @patch('torch.cuda.is_available')
    def test_clear_gpu_cache_no_cuda(self, mock_cuda_available):
        """Test clear_gpu_cache when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        
        clear_gpu_cache()  # Should not raise an error
    
    @patch('torch.cuda.is_available')
    @patch('src.utils.model_utils.get_model_memory_usage')
    def test_get_device_info_cuda(self, mock_memory_usage, mock_cuda_available):
        """Test get_device_info when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_memory_usage.return_value = {"memory_test": "value"}
        
        with patch('torch.__version__', '2.0.0'), \
             patch('torch.version.cuda', '12.0'), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.get_device_name', return_value='Test GPU'):
            
            result = get_device_info()
        
        self.assertEqual(result["torch_version"], "2.0.0")
        self.assertTrue(result["cuda_available"])
        self.assertEqual(result["cuda_version"], "12.0")
        self.assertEqual(result["device_count"], 1)
        self.assertEqual(result["current_device"], 0)
        self.assertEqual(result["device_name"], "Test GPU")
        self.assertIn("memory_test", result)
    
    @patch('torch.cuda.is_available')
    def test_get_device_info_no_cuda(self, mock_cuda_available):
        """Test get_device_info when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        
        with patch('torch.__version__', '2.0.0'):
            result = get_device_info()
        
        self.assertEqual(result["torch_version"], "2.0.0")
        self.assertFalse(result["cuda_available"])
        self.assertEqual(result["device"], "cpu")
    
    def test_validate_model_path_nonexistent(self):
        """Test validate_model_path with non-existent path."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = validate_model_path(nonexistent_path)
        
        self.assertFalse(result)
    
    def test_validate_model_path_not_directory(self):
        """Test validate_model_path with file instead of directory."""
        test_file = os.path.join(self.temp_dir, "test_file.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        with patch('sys.stdout'):  # Suppress progress messages
            result = validate_model_path(test_file)
        
        self.assertFalse(result)
    
    def test_validate_model_path_valid_directory(self):
        """Test validate_model_path with valid directory."""
        # Create config.json file
        config_path = os.path.join(self.test_model_path, "config.json")
        with open(config_path, 'w') as f:
            f.write('{"test": "config"}')
        
        # Create model file
        model_path = os.path.join(self.test_model_path, "pytorch_model.bin")
        with open(model_path, 'wb') as f:
            f.write(b"dummy model data")
        
        result = validate_model_path(self.test_model_path)
        
        self.assertTrue(result)
    
    def test_validate_model_path_missing_files(self):
        """Test validate_model_path with directory missing expected files."""
        with patch('sys.stdout'):  # Suppress progress messages
            result = validate_model_path(self.test_model_path)
        
        # Should still return True but with warnings
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()