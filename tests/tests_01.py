import unittest
import os
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from scripts.charts import count_figures, process_files as process_charts
from scripts.keywordCloud import extract_abstract, process as process_cloud
from scripts.list import extract_links, should_include_link, clean_link

class TestAIOpenScience(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
        # Create sample XML content
        self.sample_xml = '''<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
            <teiHeader>
                <abstract>
                    <p>This is a test abstract with keywords test analysis research</p>
                </abstract>
            </teiHeader>
            <text>
                <body>
                    <figure/>
                    <figure/>
                    <p>Visit <ref target="http://example.com">link</ref></p>
                    <div type="references">
                        <p>Reference <ref target="http://reference.com">link</ref></p>
                    </div>
                </body>
            </text>
        </TEI>
        '''
        
        # Create a test XML file
        self.test_file = os.path.join(self.test_dir, "test.xml")
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write(self.sample_xml)

    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_count_figures(self):
        """Test if figure counting works correctly"""
        figure_count = count_figures(self.test_file)
        self.assertEqual(figure_count, 2, "Should find exactly 2 figures in the test XML")

    def test_extract_abstract(self):
        """Test if abstract extraction works correctly"""
        abstract = extract_abstract(self.test_file)
        expected = "This is a test abstract with keywords test analysis research "
        self.assertEqual(abstract, expected, "Abstract text should match exactly")

    def test_extract_links(self):
        """Test if link extraction works correctly"""
        links = extract_links(self.test_file)
        self.assertEqual(len(links), 1, "Should find exactly 1 non-reference link")
        self.assertEqual(links[0], "http://example.com", "Should extract correct link URL")

    def test_should_include_link(self):
        """Test link filtering logic"""
        self.assertTrue(should_include_link("http://example.com"))
        self.assertFalse(should_include_link("http://grobid.com"))
        self.assertFalse(should_include_link("#internal-ref"))
        self.assertFalse(should_include_link(""))

    def test_clean_link(self):
        """Test link cleaning functionality"""
        self.assertEqual(clean_link("http://example.com."), "http://example.com")
        self.assertEqual(clean_link("http://example.com,"), "http://example.com")
        self.assertEqual(clean_link("http://example.com"), "http://example.com")

    @patch('matplotlib.pyplot.savefig')
    def test_charts_generation(self, mock_savefig):
        """Test if charts are generated without errors"""
        process_charts(self.test_file, self.output_dir)
        mock_savefig.assert_called_once()
        
    @patch('matplotlib.pyplot.savefig')
    def test_wordcloud_generation(self, mock_savefig):
        """Test if word cloud is generated without errors"""
        process_cloud(self.test_file, self.output_dir)
        mock_savefig.assert_called_once()

    def test_multiple_files_processing(self):
        """Test processing multiple XML files"""
        # Create a second test file
        second_file = os.path.join(self.test_dir, "test2.xml")
        with open(second_file, "w", encoding="utf-8") as f:
            f.write(self.sample_xml)
            
        files = os.listdir(self.test_dir)
        self.assertEqual(len(files), 2, "Should have exactly 2 test files")

    def test_invalid_xml(self):
        """Test handling of invalid XML files"""
        invalid_file = os.path.join(self.test_dir, "invalid.xml")
        with open(invalid_file, "w") as f:
            f.write("This is not valid XML")
            
        with self.assertRaises(ET.ParseError):
            count_figures(invalid_file)

    def test_empty_directory(self):
        """Test processing an empty directory"""
        empty_dir = tempfile.mkdtemp()
        process_charts(empty_dir, self.output_dir)  # Should not raise any errors
        os.rmdir(empty_dir)

if __name__ == '__main__':
    unittest.main()
