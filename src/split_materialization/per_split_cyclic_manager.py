"""
Per-Split Cyclic Manager

Manages cyclic collections for individual splits using the same pattern as CyclicPipelineManager:
- split_X_input (current data)
- split_X_output (transformed data)
- SWAP: Drop input, Rename output -> input

Pattern for each split:
  split_X_input -> Process -> split_X_output -> SWAP -> split_X_input
  
Swap logic (identical to CyclicPipelineManager):
  1. Drop split_X_input (old data discarded)
  2. Rename split_X_output -> split_X_input (new data becomes input)
"""

from pymongo import MongoClient
from typing import List, Optional
from src.utils.logging import logger


class PerSplitCyclicManager:
    """
    Manages cyclic collections for each split independently.
    
    Simplified pattern for each split:
      split_X_input -> Process -> split_X_output -> SWAP -> split_X_input
    """
    
    def __init__(self, mongo_uri: str, db_name: str):
        """
        Initialize per-split cyclic manager.
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
    
    def get_split_ids(self) -> List[int]:
        """
        Discover all split IDs by looking for split_X_input collections.
        
        Returns:
            List of split IDs found
        """
        all_collections = self.db.list_collection_names()
        
        split_ids = []
        for coll_name in all_collections:
            if coll_name.startswith("split_") and coll_name.endswith("_input"):
                # Extract split ID: "split_0_input" -> 0
                split_id_str = coll_name.replace("split_", "").replace("_input", "")
                try:
                    split_id = int(split_id_str)
                    split_ids.append(split_id)
                except ValueError:
                    continue
        
        return sorted(split_ids)
    
    def validate_split_input_exists(self, split_id: int) -> bool:
        """
        Check if split_X_input collection exists and has data.
        
        Args:
            split_id: Split ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        input_coll = f"split_{split_id}_input"
        
        if input_coll not in self.db.list_collection_names():
            logger(f"Collection {input_coll} does not exist!", "ERROR")
            return False
        
        count = self.db[input_coll].count_documents({})
        if count == 0:
            logger(f"Collection {input_coll} is empty!", "ERROR")
            return False
        
        logger(f"Collection {input_coll} exists with {count:,} documents", "INFO")
        return True
    
    def prepare_split_for_processing(self, split_id: int, force: bool = False):
        """
        Prepare split for transformation processing.
        
        Clears split_X_output if it exists.
        
        Args:
            split_id: Split ID to prepare
            force: If True, drop existing split_X_output
        """
        output_coll = f"split_{split_id}_output"
        
        if output_coll in self.db.list_collection_names():
            if force:
                logger(f"Dropping existing {output_coll}", "INFO")
                self.db[output_coll].drop()
            else:
                count = self.db[output_coll].count_documents({})
                if count > 0:
                    raise ValueError(
                        f"{output_coll} already exists with {count} documents. "
                        f"Use force=True to overwrite."
                    )
        
        logger(f"Split {split_id} ready for processing", "INFO")
    
    def swap_split_to_input(self, split_id: int):
        """
        Swap split_X_output -> split_X_input.
        
        Uses the exact same pattern as CyclicPipelineManager:
        1. Drop input (old data discarded)
        2. Rename output -> input (new data becomes input)
        
        Args:
            split_id: Split ID to swap
        """
        input_coll = f"split_{split_id}_input"
        output_coll = f"split_{split_id}_output"
        
        logger("=" * 80, "INFO")
        logger(f"SWAPPING SPLIT {split_id} COLLECTIONS", "INFO")
        logger("=" * 80, "INFO")
        
        # Check prerequisites
        if not input_coll in self.db.list_collection_names():
            raise ValueError(f"Input collection '{input_coll}' does not exist!")
        
        if not output_coll in self.db.list_collection_names():
            raise ValueError(f"Output collection '{output_coll}' does not exist!")
        
        input_count = self.db[input_coll].count_documents({})
        output_count = self.db[output_coll].count_documents({})
        
        if output_count == 0:
            raise ValueError(f"Output collection '{output_coll}' is empty, cannot swap!")
        
        logger(f"Before swap:", "INFO")
        logger(f"  {input_coll}:  {input_count:,} documents", "INFO")
        logger(f"  {output_coll}: {output_count:,} documents", "INFO")
        
        # Step 1: Drop input
        logger(f"Step 1: Dropping '{input_coll}'", "INFO")
        self.db[input_coll].drop()
        logger(f"Dropped '{input_coll}'", "INFO")
        
        # Step 2: Rename output -> input
        logger(f"Step 2: Renaming '{output_coll}' -> '{input_coll}'", "INFO")
        self.db[output_coll].rename(input_coll, dropTarget=False)
        logger(f"Renamed '{output_coll}' -> '{input_coll}'", "INFO")
        
        # Verify
        new_input_count = self.db[input_coll].count_documents({})
        logger(f"After swap:", "INFO")
        logger(f"  {input_coll}: {new_input_count:,} documents", "INFO")
        
        if new_input_count != output_count:
            raise ValueError(
                f"Swap verification failed! Expected={output_count}, Got={new_input_count}"
            )
        
        logger("=" * 80, "INFO")
        logger(f"SWAP COMPLETED SUCCESSFULLY", "INFO")
        logger("=" * 80, "INFO")
        
        return True
    
    def get_split_state(self, split_id: int) -> dict:
        """
        Get current state of a split's collections.
        
        Args:
            split_id: Split ID to check
            
        Returns:
            Dictionary with collection states
        """
        input_coll = f"split_{split_id}_input"
        output_coll = f"split_{split_id}_output"
        
        all_collections = self.db.list_collection_names()
        
        state = {
            "split_id": split_id,
            "input": {
                "exists": input_coll in all_collections,
                "count": self.db[input_coll].count_documents({}) if input_coll in all_collections else 0
            },
            "output": {
                "exists": output_coll in all_collections,
                "count": self.db[output_coll].count_documents({}) if output_coll in all_collections else 0
            }
        }
        
        return state
    
    def print_split_state(self, split_id: int):
        """Print current state of a split."""
        state = self.get_split_state(split_id)
        
        logger(f"Split {split_id} State:", "INFO")
        logger(f"  split_{split_id}_input:  "
              f"{'EXISTS' if state['input']['exists'] else 'MISSING'} "
              f"({state['input']['count']:,} docs)", "INFO")
        logger(f"  split_{split_id}_output: "
              f"{'EXISTS' if state['output']['exists'] else 'MISSING'} "
              f"({state['output']['count']:,} docs)", "INFO")
    
    def print_all_splits_state(self):
        """Print state of all discovered splits."""
        split_ids = self.get_split_ids()
        
        if not split_ids:
            logger("No split_X_input collections found!", "WARNING")
            return
        
        logger("=" * 60, "INFO")
        logger(f"SPLIT COLLECTIONS STATE ({len(split_ids)} splits found)", "INFO")
        logger("=" * 60, "INFO")
        
        for split_id in split_ids:
            self.print_split_state(split_id)
            logger("", "INFO")
    
    def archive_current_input(self, split_id: int, archive_suffix: str = "_archive"):
        """
        Archive current split_X_input before transformation.
        
        Optional: Use this to keep a backup of original data.
        
        Args:
            split_id: Split ID to archive
            archive_suffix: Suffix for archive collection (default: "_archive")
        """
        input_coll = f"split_{split_id}_input"
        archive_coll = f"split_{split_id}{archive_suffix}"
        
        if input_coll not in self.db.list_collection_names():
            logger(f"{input_coll} doesn't exist, nothing to archive", "WARNING")
            return
        
        # Drop existing archive if present
        if archive_coll in self.db.list_collection_names():
            logger(f"Dropping existing {archive_coll}", "INFO")
            self.db[archive_coll].drop()
        
        # Copy input to archive (not rename, so input remains)
        logger(f"Archiving {input_coll} -> {archive_coll}", "INFO")
        
        # MongoDB doesn't have native copy, so we use aggregate $out
        self.db[input_coll].aggregate([
            {"$match": {}},
            {"$out": archive_coll}
        ])
        
        count = self.db[archive_coll].count_documents({})
        logger(f"Archived {count:,} documents to {archive_coll}", "INFO")
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()