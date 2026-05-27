package app.locallexis.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [
        TranscriptEntity::class,
        SegmentEntity::class,
        SegmentFtsEntity::class,
        SpeakerEntity::class,
        DeviceEntity::class,
        SyncStateEntity::class,
    ],
    version = 1,
    exportSchema = true,
)
abstract class LocalLexisDatabase : RoomDatabase() {
    abstract fun transcriptDao(): TranscriptDao
    abstract fun segmentDao(): SegmentDao
    abstract fun speakerDao(): SpeakerDao
    abstract fun deviceDao(): DeviceDao
    abstract fun syncStateDao(): SyncStateDao
    abstract fun searchDao(): SearchDao

    companion object {
        @Volatile
        private var instance: LocalLexisDatabase? = null

        fun get(context: Context): LocalLexisDatabase =
            instance ?: synchronized(this) {
                instance ?: build(context).also { instance = it }
            }

        private fun build(context: Context): LocalLexisDatabase =
            Room.databaseBuilder(
                context.applicationContext,
                LocalLexisDatabase::class.java,
                "locallexis.db",
            ).build()
    }
}
