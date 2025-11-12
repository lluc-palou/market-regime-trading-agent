from pymongo import MongoClient
from src.utils.logging import logger
from typing import Optional, Dict, Any, List
from pymongo.errors import CollectionInvalid, OperationFailure

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CyclicPipelineManager:
    """
    Manages cyclic pipeline collection operations.
    
    The cyclic pipeline uses 3 collections:
    - raw_lob: Permanent archive (source of truth)
    - lob_input: Working input (read by current stage)
    - lob_output: Working output (written by current stage)
    
    Workflow:
    1. Initialize: raw_lob -> lob_input
    2. Stage: lob_input -> process -> lob_output
    3. Swap: lob_output -> lob_input (prepare for next stage)
    4. Repeat steps 2-3 for each stage
    """
    
    # Collection names
    ARCHIVE_COLLECTION = "raw_lob"
    INPUT_COLLECTION = "input"
    OUTPUT_COLLECTION = "output"
    
    # Pipeline stages
    STAGES = {
        3: "data_splitting",
        4: "feature_engineering",
        5: "lob_standardization",
        6: "materialize_splits"
    }
    
    def __init__(self, db_name: str, mongo_uri: str = "mongodb://127.0.0.1:27017/"):
        """
        Initialize pipeline manager.
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        
        logger("Pipeline manager initialized", "INFO")
    
    def __del__(self):
        """Close MongoDB connection on cleanup."""
        if hasattr(self, 'client'):
            self.client.close()
    
    # =============================================================================
    # Collection Existence Checks
    # =============================================================================
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the database.
        
        Args:
            collection_name: Name of collection to check
            
        Returns:
            True if collection exists, False otherwise
        """
        return collection_name in self.db.list_collection_names()
    
    def get_collection_count(self, collection_name: str) -> int:
        """
        Get document count in a collection.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Number of documents, or 0 if collection doesn't exist
        """
        if not self.collection_exists(collection_name):
            return 0
        return self.db[collection_name].count_documents({})
    
    def get_collection_size_mb(self, collection_name: str) -> float:
        """
        Get collection size in MB.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Size in MB, or 0 if collection doesn't exist
        """
        if not self.collection_exists(collection_name):
            return 0.0
        
        stats = self.db.command("collstats", collection_name)
        size_bytes = stats.get("size", 0)
        return size_bytes / (1024 * 1024)
    
    # =============================================================================
    # Collection Operations
    # =============================================================================
    
    def drop_collection(self, collection_name: str, safe: bool = True) -> bool:
        """
        Drop a collection.
        
        Args:
            collection_name: Name of collection to drop
            safe: If True, protect archive collection from deletion
            
        Returns:
            True if dropped, False if didn't exist
            
        Raises:
            ValueError: If trying to drop archive with safe=True
        """
        # Protect archive collection
        if safe and collection_name == self.ARCHIVE_COLLECTION:
            raise ValueError(f"Cannot drop archive collection '{self.ARCHIVE_COLLECTION}' with safe=True")
        
        if not self.collection_exists(collection_name):
            logger(f"Collection '{collection_name}' does not exist, nothing to drop", "WARNING")
            return False
        
        count = self.get_collection_count(collection_name)
        size_mb = self.get_collection_size_mb(collection_name)
        
        logger(f"Dropping collection '{collection_name}' ({count:,} docs, {size_mb:.2f} MB)", "INFO")
        self.db[collection_name].drop()
        logger(f"Collection '{collection_name}' dropped", "INFO")
        
        return True
    
    def rename_collection(self, old_name: str, new_name: str, 
                         drop_target: bool = True) -> bool:
        """
        Rename a collection.
        
        Args:
            old_name: Current collection name
            new_name: New collection name
            drop_target: If True, drop target collection if it exists
            
        Returns:
            True if renamed successfully
            
        Raises:
            ValueError: If source doesn't exist or target exists (when drop_target=False)
        """
        if not self.collection_exists(old_name):
            raise ValueError(f"Source collection '{old_name}' does not exist")
        
        if self.collection_exists(new_name):
            if drop_target:
                logger(f"Target collection '{new_name}' exists, dropping first", "WARNING")
                self.drop_collection(new_name, safe=False)
            else:
                raise ValueError(f"Target collection '{new_name}' already exists")
        
        count = self.get_collection_count(old_name)
        logger(f"Renaming '{old_name}' -> '{new_name}' ({count:,} docs)", "INFO")
        
        self.db[old_name].rename(new_name, dropTarget=drop_target)
        logger(f"Collection renamed successfully", "INFO")
        
        return True
    
    def copy_collection(self, source: str, target: str, 
                       drop_target: bool = True) -> bool:
        """
        Copy a collection using aggregation $out.
        
        Args:
            source: Source collection name
            target: Target collection name
            drop_target: If True, drop target collection if it exists
            
        Returns:
            True if copied successfully
            
        Raises:
            ValueError: If source doesn't exist
        """
        if not self.collection_exists(source):
            raise ValueError(f"Source collection '{source}' does not exist")
        
        if self.collection_exists(target):
            if drop_target:
                logger(f"Target collection '{target}' exists, dropping first", "WARNING")
                self.drop_collection(target, safe=False)
            else:
                raise ValueError(f"Target collection '{target}' already exists")
        
        source_count = self.get_collection_count(source)
        source_size = self.get_collection_size_mb(source)
        
        logger(f"Copying '{source}' -> '{target}' ({source_count:,} docs, {source_size:.2f} MB)", "INFO")
        
        # Use aggregation $out for efficient copy
        self.db[source].aggregate([{"$out": target}])
        
        target_count = self.get_collection_count(target)
        logger(f"Copy completed: {target_count:,} documents", "INFO")
        
        return True
    
    # =============================================================================
    # Pipeline State Management
    # =============================================================================
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """
        Get current pipeline state.
        
        Returns:
            Dictionary with pipeline state information:
            - collections: Dict of collection existence and stats
            - can_initialize: Whether pipeline can be initialized
            - can_swap: Whether collections can be swapped
            - current_stage: Estimated current stage based on state
        """
        state = {
            "collections": {},
            "can_initialize": False,
            "can_swap": False,
            "current_stage": None
        }
        
        # Check each collection
        for coll in [self.ARCHIVE_COLLECTION, self.INPUT_COLLECTION, self.OUTPUT_COLLECTION]:
            if self.collection_exists(coll):
                state["collections"][coll] = {
                    "exists": True,
                    "count": self.get_collection_count(coll),
                    "size_mb": self.get_collection_size_mb(coll)
                }
            else:
                state["collections"][coll] = {
                    "exists": False,
                    "count": 0,
                    "size_mb": 0.0
                }
        
        # Determine capabilities
        archive_exists = state["collections"][self.ARCHIVE_COLLECTION]["exists"]
        input_exists = state["collections"][self.INPUT_COLLECTION]["exists"]
        output_exists = state["collections"][self.OUTPUT_COLLECTION]["exists"]
        
        state["can_initialize"] = archive_exists and not input_exists
        state["can_swap"] = input_exists and output_exists
        
        # Estimate current stage
        if not input_exists and not output_exists:
            state["current_stage"] = "uninitialized"
        elif input_exists and not output_exists:
            state["current_stage"] = "ready_for_processing"
        elif input_exists and output_exists:
            state["current_stage"] = "ready_for_swap"
        else:
            state["current_stage"] = "unknown"
        
        return state
    
    def print_pipeline_state(self):
        """Print current pipeline state in a readable format."""
        state = self.get_pipeline_state()
        
        logger("=" * 80, "INFO")
        logger("PIPELINE STATE", "INFO")
        logger("=" * 80, "INFO")
        
        for coll_name, coll_info in state["collections"].items():
            if coll_info["exists"]:
                logger(f"[OK] {coll_name:20s} | {coll_info['count']:>10,} docs | {coll_info['size_mb']:>8.2f} MB", "INFO")
            else:
                logger(f"[FAIL] {coll_name:20s} | Does not exist", "INFO")
        
        logger("-" * 80, "INFO")
        logger(f"Current Stage: {state['current_stage']}", "INFO")
        logger(f"Can Initialize: {state['can_initialize']}", "INFO")
        logger(f"Can Swap: {state['can_swap']}", "INFO")
        logger("=" * 80, "INFO")
    
    # =============================================================================
    # Pipeline Operations
    # =============================================================================
    
    def initialize_pipeline(self, force: bool = False) -> bool:
        """
        Initialize cyclic pipeline by copying archive to input.
        
        Args:
            force: If True, drop existing working collections before initializing
            
        Returns:
            True if initialized successfully
            
        Raises:
            ValueError: If prerequisites not met
        """
        logger("=" * 80, "INFO")
        logger("INITIALIZING CYCLIC PIPELINE", "INFO")
        logger("=" * 80, "INFO")
        
        # Check archive exists
        if not self.collection_exists(self.ARCHIVE_COLLECTION):
            raise ValueError(f"Archive collection '{self.ARCHIVE_COLLECTION}' does not exist!")
        
        archive_count = self.get_collection_count(self.ARCHIVE_COLLECTION)
        archive_size = self.get_collection_size_mb(self.ARCHIVE_COLLECTION)
        
        if archive_count == 0:
            raise ValueError(f"Archive collection '{self.ARCHIVE_COLLECTION}' is empty!")
        
        logger(f"Archive: {archive_count:,} documents, {archive_size:.2f} MB", "INFO")
        
        # Check if working collections already exist
        if self.collection_exists(self.INPUT_COLLECTION):
            if force:
                logger(f"Force mode: dropping existing '{self.INPUT_COLLECTION}'", "WARNING")
                self.drop_collection(self.INPUT_COLLECTION, safe=False)
            else:
                raise ValueError(f"Input collection '{self.INPUT_COLLECTION}' already exists! Use force=True to override")
        
        if self.collection_exists(self.OUTPUT_COLLECTION):
            if force:
                logger(f"Force mode: dropping existing '{self.OUTPUT_COLLECTION}'", "WARNING")
                self.drop_collection(self.OUTPUT_COLLECTION, safe=False)
            else:
                raise ValueError(f"Output collection '{self.OUTPUT_COLLECTION}' already exists! Use force=True to override")
        
        # Copy archive to input
        logger("Copying archive -> input...", "INFO")
        self.copy_collection(self.ARCHIVE_COLLECTION, self.INPUT_COLLECTION, drop_target=False)
        
        # Verify
        input_count = self.get_collection_count(self.INPUT_COLLECTION)
        if input_count != archive_count:
            raise ValueError(f"Copy verification failed! Archive={archive_count}, Input={input_count}")
        
        logger("=" * 80, "INFO")
        logger("PIPELINE INITIALIZED SUCCESSFULLY", "INFO")
        logger("=" * 80, "INFO")
        
        self.print_pipeline_state()
        
        return True
    
    def swap_working_collections(self) -> bool:
        """
        Swap working collections: output -> input.
        
        This prepares the pipeline for the next stage by:
        1. Dropping current input (no longer needed)
        2. Renaming output to input
        
        Returns:
            True if swapped successfully
            
        Raises:
            ValueError: If prerequisites not met
        """
        logger("=" * 80, "INFO")
        logger("SWAPPING WORKING COLLECTIONS", "INFO")
        logger("=" * 80, "INFO")
        
        # Check prerequisites
        if not self.collection_exists(self.INPUT_COLLECTION):
            raise ValueError(f"Input collection '{self.INPUT_COLLECTION}' does not exist!")
        
        if not self.collection_exists(self.OUTPUT_COLLECTION):
            raise ValueError(f"Output collection '{self.OUTPUT_COLLECTION}' does not exist!")
        
        input_count = self.get_collection_count(self.INPUT_COLLECTION)
        output_count = self.get_collection_count(self.OUTPUT_COLLECTION)
        
        logger(f"Before swap:", "INFO")
        logger(f"  Input:  {input_count:,} documents", "INFO")
        logger(f"  Output: {output_count:,} documents", "INFO")
        
        # Drop input
        logger(f"Step 1: Dropping '{self.INPUT_COLLECTION}'", "INFO")
        self.drop_collection(self.INPUT_COLLECTION, safe=False)
        
        # Rename output to input
        logger(f"Step 2: Renaming '{self.OUTPUT_COLLECTION}' -> '{self.INPUT_COLLECTION}'", "INFO")
        self.rename_collection(self.OUTPUT_COLLECTION, self.INPUT_COLLECTION, drop_target=False)
        
        # Verify
        new_input_count = self.get_collection_count(self.INPUT_COLLECTION)
        logger(f"After swap:", "INFO")
        logger(f"  Input:  {new_input_count:,} documents", "INFO")
        
        if new_input_count != output_count:
            raise ValueError(f"Swap verification failed! Expected={output_count}, Got={new_input_count}")
        
        logger("=" * 80, "INFO")
        logger("SWAP COMPLETED SUCCESSFULLY", "INFO")
        logger("=" * 80, "INFO")
        
        self.print_pipeline_state()
        
        return True
    
    def cleanup_working_collections(self, keep_archive: bool = True) -> bool:
        """
        Clean up working collections.
        
        Args:
            keep_archive: If True, preserve archive collection (recommended)
            
        Returns:
            True if cleanup successful
        """
        logger("=" * 80, "INFO")
        logger("CLEANING UP WORKING COLLECTIONS", "INFO")
        logger("=" * 80, "INFO")
        
        # Drop input
        if self.collection_exists(self.INPUT_COLLECTION):
            self.drop_collection(self.INPUT_COLLECTION, safe=False)
        
        # Drop output
        if self.collection_exists(self.OUTPUT_COLLECTION):
            self.drop_collection(self.OUTPUT_COLLECTION, safe=False)
        
        # Optionally drop archive (not recommended)
        if not keep_archive:
            logger("WARNING: Dropping archive collection!", "WARNING")
            if self.collection_exists(self.ARCHIVE_COLLECTION):
                self.drop_collection(self.ARCHIVE_COLLECTION, safe=False)
        
        logger("=" * 80, "INFO")
        logger("CLEANUP COMPLETED", "INFO")
        logger("=" * 80, "INFO")
        
        self.print_pipeline_state()
        
        return True
    
    def reset_to_archive(self, force: bool = False) -> bool:
        """
        Reset pipeline to archive state.
        
        Drops working collections and re-initializes from archive.
        Useful for restarting pipeline from scratch.
        
        Args:
            force: If True, proceed even if working collections exist
            
        Returns:
            True if reset successful
        """
        logger("=" * 80, "INFO")
        logger("RESETTING PIPELINE TO ARCHIVE", "INFO")
        logger("=" * 80, "INFO")
        
        # Drop working collections
        if self.collection_exists(self.INPUT_COLLECTION):
            logger(f"Dropping '{self.INPUT_COLLECTION}'", "INFO")
            self.drop_collection(self.INPUT_COLLECTION, safe=False)
        
        if self.collection_exists(self.OUTPUT_COLLECTION):
            logger(f"Dropping '{self.OUTPUT_COLLECTION}'", "INFO")
            self.drop_collection(self.OUTPUT_COLLECTION, safe=False)
        
        # Re-initialize
        logger("Re-initializing from archive", "INFO")
        self.initialize_pipeline(force=force)
        
        logger("=" * 80, "INFO")
        logger("RESET COMPLETED", "INFO")
        logger("=" * 80, "INFO")
        
        return True
    
    # =============================================================================
    # Validation
    # =============================================================================
    
    def validate_can_run_stage(self, stage_number: int) -> bool:
        """
        Validate that prerequisites are met for running a stage.
        
        Args:
            stage_number: Stage number (3, 4, 5, or 6)
            
        Returns:
            True if can run stage
            
        Raises:
            ValueError: If prerequisites not met
        """
        stage_name = self.STAGES.get(stage_number, f"stage_{stage_number}")
        
        logger(f"Validating prerequisites for Stage {stage_number}: {stage_name}", "INFO")
        
        # All stages require input collection to exist
        if not self.collection_exists(self.INPUT_COLLECTION):
            raise ValueError(f"Cannot run stage: '{self.INPUT_COLLECTION}' does not exist. Run initialization first.")
        
        # Check input has data
        input_count = self.get_collection_count(self.INPUT_COLLECTION)
        if input_count == 0:
            raise ValueError(f"Cannot run stage: '{self.INPUT_COLLECTION}' is empty")
        
        # Output should NOT exist (clean slate for stage)
        if self.collection_exists(self.OUTPUT_COLLECTION):
            raise ValueError(
                f"Cannot run stage: '{self.OUTPUT_COLLECTION}' already exists. "
                f"Run swap or drop it first."
            )
        
        logger(f"[OK] Prerequisites met for Stage {stage_number}", "INFO")
        logger(f"  Input: {input_count:,} documents ready for processing", "INFO")
        
        return True