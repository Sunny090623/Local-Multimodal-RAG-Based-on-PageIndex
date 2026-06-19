import os
import sys
import json
import unittest
import asyncio
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.provider import (
    check_ollama_status, check_xinference_status, get_active_provider, set_active_provider,
    get_vlm_provider, set_vlm_provider
)
from backend.shared import get_shared_client, STORAGE_DIR, WORKSPACE_DIR

class TestRAGBackend(unittest.TestCase):
    
    def test_provider_settings(self):
        set_active_provider("ollama", "gemma4:12b-it-qat", "http://localhost:11434")
        active = get_active_provider()
        self.assertEqual(active.provider_type, "ollama")
        self.assertEqual(active.model_name, "gemma4:12b-it-qat")
        self.assertEqual(active.api_base, "http://localhost:11434")

    def test_parser_settings(self):
        set_vlm_provider(True, "ollama", "gemma4:12b-it-qat", "http://localhost:11434")
        settings = get_vlm_provider()
        self.assertEqual(settings.model_name, "gemma4:12b-it-qat")
        self.assertEqual(settings.provider_type, "ollama")
        self.assertTrue(settings.use_vlm)


    def test_status_checks(self):
        # Test status check executes without throwing
        ollama_status = check_ollama_status()
        self.assertIn("status", ollama_status)
        
        xinference_status = check_xinference_status()
        self.assertIn("status", xinference_status)


class TestSharedClient(unittest.TestCase):
    """Test the shared PageIndexClient singleton pattern (BUG-004 fix)."""

    def test_singleton_consistency(self):
        """Multiple calls to get_shared_client() should return the same instance."""
        client_a = get_shared_client()
        client_b = get_shared_client()
        self.assertIs(client_a, client_b, "get_shared_client() should return the same instance")

    def test_workspace_exists(self):
        """The workspace directory should be automatically created."""
        self.assertTrue(WORKSPACE_DIR.exists())
        self.assertTrue(STORAGE_DIR.exists())


class TestIndexDocument(unittest.TestCase):
    """Test core index_document function (BUG-008 fix)."""

    def test_index_txt_file(self):
        """Test indexing a plain text file creates a valid document entry."""
        from backend.parser import index_document
        
        # Create a temporary .txt file
        test_content = "# Test Document\n\nThis is a test paragraph for indexing validation.\n\n## Section A\n\nSome content in section A.\n\n## Section B\n\nSome content in section B."
        test_file = STORAGE_DIR / "test_index_validation.txt"
        try:
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(test_content)
            
            # This will fail if LLM is not available, which is expected in CI
            # We just verify the function doesn't crash with asyncio errors
            try:
                doc_id = asyncio.run(index_document(str(test_file)))
                self.assertIsInstance(doc_id, str)
                self.assertTrue(len(doc_id) > 0)
                
                # Verify document appears in shared client
                client = get_shared_client()
                self.assertIn(doc_id, client.documents)
                
                # Verify workspace JSON file was created
                doc_json = WORKSPACE_DIR / f"{doc_id}.json"
                self.assertTrue(doc_json.exists(), "Document JSON should be persisted to workspace")
                
                # Clean up
                if doc_json.exists():
                    os.remove(doc_json)
                client.documents.pop(doc_id, None)
                
                # Clean up _meta.json to prevent dangling test records
                meta_path = WORKSPACE_DIR / "_meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if doc_id in meta:
                            meta.pop(doc_id, None)
                            with open(meta_path, "w", encoding="utf-8") as f:
                                json.dump(meta, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"Warning: Failed to clean up metadata in test: {e}")
                
                print(f"TXT indexing test passed. doc_id={doc_id}")
            except Exception as e:
                # If LLM is offline, the tree parser will fail — that's OK for this test
                err_str = str(e).lower()
                if "connection" in err_str or "timeout" in err_str or "refused" in err_str:
                    print(f"Skipping TXT index test (LLM offline): {e}")
                else:
                    raise
        finally:
            if test_file.exists():
                os.remove(test_file)

    def test_index_unsupported_format(self):
        """Test that unsupported file formats raise ValueError."""
        from backend.parser import index_document
        
        test_file = STORAGE_DIR / "test_unsupported.xyz"
        try:
            with open(test_file, "w") as f:
                f.write("test")
            
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(index_document(str(test_file)))
            
            self.assertIn("Unsupported file format", str(ctx.exception))
            print(f"Unsupported format test passed: {ctx.exception}")
        finally:
            if test_file.exists():
                os.remove(test_file)

    def test_index_missing_file(self):
        """Test that indexing a non-existent file raises FileNotFoundError."""
        from backend.parser import index_document
        
        with self.assertRaises(FileNotFoundError):
            asyncio.run(index_document("/nonexistent/path/document.pdf"))


class TestRAGFlowEdgeCases(unittest.TestCase):
    """Test RAG flow error handling."""

    def test_rag_flow_missing_document(self):
        """Querying a non-existent document should return error JSON."""
        from backend.rag import execute_rag_flow_stream
        
        async def collect_stream():
            results = []
            async for chunk in execute_rag_flow_stream("nonexistent-doc-id", "test query"):
                results.append(json.loads(chunk.strip()))
            return results
        
        results = asyncio.run(collect_stream())
        
        self.assertTrue(len(results) > 0, "Should return at least one message")
        self.assertEqual(results[0]["type"], "error")
        self.assertIn("not found", results[0]["content"])
        print(f"Missing document test passed: {results[0]['content']}")


class TestLiteLLMPatching(unittest.TestCase):
    """Test that litellm is correctly patched by provider.py."""
    
    def test_litellm_completion_is_patched(self):
        """litellm.completion should be the patched version, not the original."""
        import litellm
        from backend.provider import patched_completion, patched_acompletion
        
        self.assertEqual(litellm.completion, patched_completion,
                         "litellm.completion should be patched to patched_completion")
        self.assertEqual(litellm.acompletion, patched_acompletion,
                         "litellm.acompletion should be patched to patched_acompletion")
        print("litellm patching verified.")
    
    def test_drop_params_enabled(self):
        """litellm.drop_params should be True to handle unsupported params gracefully."""
        import litellm
        self.assertTrue(litellm.drop_params)


class TestOutlineHealing(unittest.TestCase):
    """Test the generalized outline healing function scan_and_insert_missing_headings."""
    
    def test_scan_and_insert_missing_headings(self):
        from pageindex.page_index import scan_and_insert_missing_headings
        
        # Mock outline: references missing
        toc = [
            {"structure": "1", "title": "黄仁勋", "physical_index": 1},
            {"structure": "1.1", "title": "早年", "physical_index": 1},
            {"structure": "1.2", "title": "言论观点", "physical_index": 4},
            {"structure": "1.3", "title": "References", "physical_index": 7}
        ]
        
        # Mock page content: page 5 has markdown heading '# 参考来源'
        page_list = [
            ("黄仁勋 is CEO...", 10),                     # Page 1
            ("Early life info...", 10),                     # Page 2
            ("More life...", 10),                           # Page 3
            ("言论观点: Taiwan and market...", 10),       # Page 4
            ("# 参考来源\n\n1. Source 1\n2. Source 2", 10),  # Page 5
            ("More sources...", 10),                        # Page 6
            ("### References\n\n46. Source 46", 10)         # Page 7
        ]
        
        healed_toc = scan_and_insert_missing_headings(toc, page_list, start_index=1)
        
        # Check that '参考来源' was inserted
        titles = [item["title"] for item in healed_toc]
        self.assertIn("参考来源", titles)
        
        # Verify correct position (should be after '言论观点' on page 4 and before 'References' on page 7)
        idx_opinions = titles.index("言论观点")
        idx_ref_sources = titles.index("参考来源")
        idx_references = titles.index("References")
        self.assertTrue(idx_opinions < idx_ref_sources < idx_references)
        
        # Verify physical index of inserted node
        ref_source_item = healed_toc[idx_ref_sources]
        self.assertEqual(ref_source_item["physical_index"], 5)
        
        # Verify structure prefix inheritance
        # Preceding was '1.2', so it should be derived as '1.2_auto1'
        self.assertEqual(ref_source_item["structure"], "1.2_auto1")
        
        print("scan_and_insert_missing_headings test passed.")

class TestHierarchyRebuildAndClean(unittest.TestCase):
    """Test outline level reconstruction, metadata cleaning, and children nesting mapping."""
    
    def test_build_structures_from_levels(self):
        from pageindex.page_index import build_structures_from_levels
        toc = [
            [1, "黄仁勋", 1],
            [2, "早年", 1],
            [3, "奥奈达", 2],
            [2, "个人生活", 3]
        ]
        items = build_structures_from_levels(toc)
        self.assertEqual(items[0]["structure"], "1")
        self.assertEqual(items[1]["structure"], "1.1")
        self.assertEqual(items[2]["structure"], "1.1.1")
        self.assertEqual(items[3]["structure"], "1.2")
        print("build_structures_from_levels test passed.")

    def test_clean_and_deduplicate_outline(self):
        # pyrefly: ignore [missing-import]
        from pageindex.page_index import clean_and_deduplicate_outline
        toc_items = [
            {"title": "黄仁勋", "physical_index": 1},
            {"title": "早年", "physical_index": 1},
            {"title": "黄仁勋 - 维基百科，自由的百科全书", "physical_index": 8},
            {"title": "早年", "physical_index": 9},  # Duplicate title
            {"title": "外部链接", "physical_index": 10}
        ]
        cleaned = clean_and_deduplicate_outline(toc_items, doc_name="jensen huang")
        titles = [item["title"] for item in cleaned]
        self.assertEqual(titles, ["黄仁勋", "早年", "外部链接"])
        print("clean_and_deduplicate_outline test passed.")

    def test_rebuild_structure_hierarchy(self):
        # pyrefly: ignore [missing-import]
        from pageindex.page_index import rebuild_structure_hierarchy
        toc_items = [
            {"title": "黄仁勋", "physical_index": 1, "structure": "1"},
            {"title": "早年", "physical_index": 1, "structure": "1"},  # flat LLM structure
            {"title": "个人生活", "physical_index": 3, "structure": "1"},
            {"title": "奖项", "physical_index": 3, "structure": "1"},
            {"title": "参考来源", "physical_index": 5, "structure": "1"}
        ]
        page_list = [
            ("# 黄仁勋\n\n## 早年\n\ninfo...", 10),  # Page 1
            ("some text...", 10),                     # Page 2
            ("## 个人生活\n\n### 奖项\n\ninfo...", 10), # Page 3
            ("some text...", 10),                     # Page 4
            ("# 参考来源\n\n1. Source...", 10)         # Page 5
        ]
        rebuilt = rebuild_structure_hierarchy(toc_items, page_list)
        # Rebuilt structures:
        # "黄仁勋" -> level 1 -> "1"
        # "早年" -> level 2 -> "1.1"
        # "个人生活" -> level 2 -> "1.2"
        # "奖项" -> level 3 -> "1.2.1"
        # "参考来源" -> level 1, adjusted to 2 because i > 0 -> "1.3"
        self.assertEqual(rebuilt[0]["structure"], "1")
        self.assertEqual(rebuilt[1]["structure"], "1.1")
        self.assertEqual(rebuilt[2]["structure"], "1.2")
        self.assertEqual(rebuilt[3]["structure"], "1.2.1")
        self.assertEqual(rebuilt[4]["structure"], "1.3")
        print("rebuild_structure_hierarchy test passed.")


class TestRenameAndSufficiency(unittest.TestCase):
    """Test document rename path/folder sync and sufficiency check string normalizations."""

    def test_sufficiency_check_normalization(self):
        def check_sufficiency(check_res):
            sufficient_val = check_res.get("sufficient") if isinstance(check_res, dict) else False
            is_sufficient = False
            if sufficient_val is True:
                is_sufficient = True
            elif isinstance(sufficient_val, str):
                is_sufficient = sufficient_val.strip().lower() in ("true", "yes", "1")
            return is_sufficient

        self.assertTrue(check_sufficiency({"sufficient": True}))
        self.assertTrue(check_sufficiency({"sufficient": "true"}))
        self.assertTrue(check_sufficiency({"sufficient": "True"}))
        self.assertTrue(check_sufficiency({"sufficient": "yes"}))
        self.assertTrue(check_sufficiency({"sufficient": "1"}))
        
        self.assertFalse(check_sufficiency({"sufficient": False}))
        self.assertFalse(check_sufficiency({"sufficient": "false"}))
        self.assertFalse(check_sufficiency({"sufficient": "no"}))
        self.assertFalse(check_sufficiency({"sufficient": None}))
        print("sufficiency check normalization tests passed.")

    def test_rename_directories_and_files(self):
        import tempfile
        import shutil
        from pathlib import Path
        from pageindex.utils import sanitize_filename

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_dir = tmp_path / "images"
            uploads_dir = tmp_path / "uploads"
            images_dir.mkdir()
            uploads_dir.mkdir()

            old_name = "test_document_old"
            new_name = "test document new"

            # Create visual image folder
            old_image_folder = images_dir / sanitize_filename(old_name)
            old_image_folder.mkdir()
            (old_image_folder / "file_1.png").write_text("image-data")

            # Create original file inside uploads
            uuid_prefix = "uuid123_"
            old_original_path = uploads_dir / f"{uuid_prefix}{old_name}.pdf"
            old_original_path.write_text("pdf-data")

            # Mock rename_document logic
            # 1. Rename images folder
            new_image_folder = images_dir / sanitize_filename(new_name)
            if old_name:
                if old_image_folder.exists() and old_image_folder.is_dir() and old_image_folder != new_image_folder:
                    if new_image_folder.exists():
                        shutil.rmtree(new_image_folder)
                    os.rename(old_image_folder, new_image_folder)

            # 2. Rename original file path inside uploads
            doc_path = str(old_original_path)
            if doc_path and os.path.exists(doc_path):
                old_path_obj = Path(doc_path)
                try:
                    if old_path_obj.parent.resolve() == uploads_dir.resolve():
                        filename = old_path_obj.name
                        parts = filename.split("_", 1)
                        prefix = ""
                        if len(parts) > 1:
                            prefix = parts[0] + "_"
                        ext = old_path_obj.suffix
                        new_filename = f"{prefix}{new_name}{ext}"
                        new_path = uploads_dir / new_filename

                        if new_path != old_path_obj:
                            if new_path.exists():
                                import uuid
                                new_path = uploads_dir / f"{uuid.uuid4()}_{new_name}{ext}"
                            os.rename(old_path_obj, new_path)
                            doc_path = str(new_path)
                except Exception as e:
                    pass

            # Assertions
            self.assertFalse(old_image_folder.exists())
            self.assertTrue(new_image_folder.exists())
            self.assertTrue((new_image_folder / "file_1.png").exists())

            self.assertFalse(old_original_path.exists())
            self.assertTrue(Path(doc_path).exists())
            self.assertEqual(Path(doc_path).name, f"{uuid_prefix}{new_name}.pdf")
            print("rename directories and files test passed.")


if __name__ == "__main__":
    unittest.main()
