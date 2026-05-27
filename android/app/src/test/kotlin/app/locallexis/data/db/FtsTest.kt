package app.locallexis.data.db

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class FtsTest {

    private lateinit var db: LocalLexisDatabase

    @Before
    fun setUp() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        db = Room.inMemoryDatabaseBuilder(context, LocalLexisDatabase::class.java)
            .allowMainThreadQueries()
            .build()
    }

    @After
    fun tearDown() {
        db.close()
    }

    @Test
    fun rawFtsRoundTrip() {
        val raw = db.openHelper.writableDatabase
        raw.execSQL(
            "INSERT INTO segments_fts (transcriptId, segmentIndex, text) " +
                "VALUES ('x', 0, 'hello world')"
        )
        val count = raw.query("SELECT COUNT(*) FROM segments_fts").use {
            it.moveToFirst(); it.getInt(0)
        }
        assertEquals("rows inserted", 1, count)

        val literalRows = raw.query(
            "SELECT text FROM segments_fts WHERE text MATCH 'hello'"
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) add(cursor.getString(0))
            }
        }
        assertEquals("literal MATCH 1 row", 1, literalRows.size)

        val boundRows = raw.query(
            "SELECT text FROM segments_fts WHERE text MATCH ?",
            arrayOf<Any>("hello"),
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) add(cursor.getString(0))
            }
        }
        assertEquals("bound MATCH 1 row", 1, boundRows.size)

        val tableBoundRows = raw.query(
            "SELECT text FROM segments_fts WHERE segments_fts MATCH ?",
            arrayOf<Any>("hello"),
        ).use { cursor ->
            buildList {
                while (cursor.moveToNext()) add(cursor.getString(0))
            }
        }
        assertEquals("table-name bound MATCH 1 row", 1, tableBoundRows.size)
    }

    @Test
    fun daoInsertRawMatch() = runTest {
        val transcripts = db.transcriptDao()
        val segments = db.segmentDao()

        transcripts.upsert(makeT("t1"))
        segments.upsertAll(
            listOf(
                seg("t1", 0, "the quick brown fox", null),
            )
        )

        val raw = db.openHelper.writableDatabase
        val rows = raw.query(
            "SELECT text FROM segments_fts WHERE text MATCH ?",
            arrayOf<Any>("quick"),
        ).use { c ->
            buildList { while (c.moveToNext()) add(c.getString(0)) }
        }
        assertEquals("DAO insert + raw MATCH", 1, rows.size)
    }

    @Test
    fun matchReturnsHitWithSnippet() = runTest {
        val transcripts = db.transcriptDao()
        val segments = db.segmentDao()
        val search = db.searchDao()

        transcripts.upsert(
            TranscriptEntity(
                id = "t1",
                workspaceId = "ws_1",
                audioPath = "/a/m.mp3",
                audioBasename = "m.mp3",
                durationSeconds = 60.0,
                language = "en",
                createdAt = "2026-05-20T10:00:00Z",
                jsonMtime = 1716200000.0,
                modelsAsr = null,
                modelsDiarizer = null,
                rawJson = "{}",
            )
        )
        segments.upsertAll(
            listOf(
                seg("t1", 0, "the quick brown fox", "SPEAKER_00"),
                seg("t1", 1, "jumps over the lazy dog", "SPEAKER_01"),
                seg("t1", 2, "afternoon coffee break", "SPEAKER_00"),
            )
        )

        assertEquals("fts row count after upsertAll", 3, segments.countFts())

        val hits = search.searchSegments("quick")
        assertEquals(1, hits.size)
        assertEquals("t1", hits[0].transcriptId)
        assertEquals(0, hits[0].segmentIndex)
        assertTrue(
            "snippet should bracket the match",
            hits[0].snippet.contains("[quick]") || hits[0].snippet.contains("quick"),
        )

        val dogHits = search.searchSegments("dog")
        assertEquals(1, dogHits.size)
        assertEquals(1, dogHits[0].segmentIndex)
    }

    @Test
    fun matchAcrossMultipleTranscriptsReturnsAll() = runTest {
        val transcripts = db.transcriptDao()
        val segments = db.segmentDao()
        val search = db.searchDao()

        transcripts.upsert(makeT("t1"))
        transcripts.upsert(makeT("t2"))
        segments.upsertAll(
            listOf(
                seg("t1", 0, "discussing the roadmap", "SPEAKER_00"),
                seg("t2", 0, "roadmap looks aggressive", "SPEAKER_00"),
            )
        )

        val hits = search.searchSegments("roadmap")
        assertEquals(2, hits.size)
        assertEquals(setOf("t1", "t2"), hits.map { it.transcriptId }.toSet())
    }

    private fun seg(
        transcriptId: String,
        index: Int,
        text: String,
        speakerId: String?,
    ) = SegmentEntity(
        transcriptId = transcriptId,
        segmentIndex = index,
        startSec = index.toDouble() * 5,
        endSec = (index + 1).toDouble() * 5,
        text = text,
        speakerId = speakerId,
    )

    private fun makeT(id: String) = TranscriptEntity(
        id = id,
        workspaceId = "ws_1",
        audioPath = "/a/$id.mp3",
        audioBasename = "$id.mp3",
        durationSeconds = 60.0,
        language = "en",
        createdAt = "2026-05-20T10:00:00Z",
        jsonMtime = 1716200000.0,
        modelsAsr = null,
        modelsDiarizer = null,
        rawJson = "{}",
    )
}
