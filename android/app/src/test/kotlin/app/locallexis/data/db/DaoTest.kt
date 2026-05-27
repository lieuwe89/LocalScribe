package app.locallexis.data.db

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class DaoTest {

    private lateinit var db: LocalLexisDatabase
    private lateinit var transcripts: TranscriptDao
    private lateinit var segments: SegmentDao
    private lateinit var speakers: SpeakerDao
    private lateinit var devices: DeviceDao
    private lateinit var syncState: SyncStateDao

    @Before
    fun setUp() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        db = Room.inMemoryDatabaseBuilder(context, LocalLexisDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        transcripts = db.transcriptDao()
        segments = db.segmentDao()
        speakers = db.speakerDao()
        devices = db.deviceDao()
        syncState = db.syncStateDao()
    }

    @After
    fun tearDown() {
        db.close()
    }

    @Test
    fun transcriptInsertAndGetById() = runTest {
        val t = TranscriptEntity(
            id = "abc123",
            workspaceId = "ws_1",
            audioPath = "/audio/meeting.mp3",
            audioBasename = "meeting.mp3",
            durationSeconds = 600.0,
            language = "nl",
            createdAt = "2026-05-20T10:00:00Z",
            jsonMtime = 1716200000.0,
            modelsAsr = "whisper-large-v3",
            modelsDiarizer = "pyannote-3.1",
            rawJson = """{"id":"abc123"}""",
        )
        transcripts.upsert(t)

        val got = transcripts.getById("abc123")
        assertNotNull(got)
        assertEquals("meeting.mp3", got!!.audioBasename)
        assertEquals("nl", got.language)
    }

    @Test
    fun transcriptListOrderedByCreatedAtDesc() = runTest {
        transcripts.upsert(makeTranscript("a", createdAt = "2026-05-20T10:00:00Z"))
        transcripts.upsert(makeTranscript("b", createdAt = "2026-05-22T10:00:00Z"))
        transcripts.upsert(makeTranscript("c", createdAt = "2026-05-21T10:00:00Z"))

        val list = transcripts.listAll().first()
        assertEquals(listOf("b", "c", "a"), list.map { it.id })
    }

    @Test
    fun segmentBySpeakerOrderedByStart() = runTest {
        transcripts.upsert(makeTranscript("t1"))
        segments.upsertAll(
            listOf(
                seg("t1", 0, 5.0, 10.0, "SPEAKER_00", "hello"),
                seg("t1", 1, 10.0, 15.0, "SPEAKER_01", "hi"),
                seg("t1", 2, 15.0, 20.0, "SPEAKER_00", "again"),
            )
        )

        val s0 = segments.bySpeaker("t1", "SPEAKER_00")
        assertEquals(listOf("hello", "again"), s0.map { it.text })
    }

    @Test
    fun speakerListForTranscript() = runTest {
        transcripts.upsert(makeTranscript("t1"))
        speakers.upsertAll(
            listOf(
                SpeakerEntity("t1", "SPEAKER_00", "Alice"),
                SpeakerEntity("t1", "SPEAKER_01", "Bob"),
            )
        )

        val list = speakers.forTranscript("t1").first()
        assertEquals(2, list.size)
        assertEquals(setOf("Alice", "Bob"), list.map { it.name }.toSet())
    }

    @Test
    fun deviceUpsertOverwritesByDeviceId() = runTest {
        devices.upsert(
            DeviceEntity(
                deviceId = "dev_1",
                name = "Pixel",
                pubkeyB64 = "AAAA",
                pairedAt = "2026-05-20T10:00:00Z",
                lastSeen = null,
            )
        )
        devices.upsert(
            DeviceEntity(
                deviceId = "dev_1",
                name = "Pixel 7",
                pubkeyB64 = "BBBB",
                pairedAt = "2026-05-20T10:00:00Z",
                lastSeen = "2026-05-25T10:00:00Z",
            )
        )

        val list = devices.listAll().first()
        assertEquals(1, list.size)
        assertEquals("Pixel 7", list[0].name)
        assertEquals("BBBB", list[0].pubkeyB64)
    }

    @Test
    fun syncStateCursorRoundTrip() = runTest {
        assertNull(syncState.getCursor("ws_1"))

        syncState.upsert(SyncStateEntity("ws_1", 1716200000.0, "2026-05-20T10:00:00Z"))
        assertEquals(1716200000.0, syncState.getCursor("ws_1")!!, 0.0001)

        syncState.upsert(SyncStateEntity("ws_1", 1716300000.0, "2026-05-21T10:00:00Z"))
        assertEquals(1716300000.0, syncState.getCursor("ws_1")!!, 0.0001)
    }

    private fun makeTranscript(
        id: String,
        createdAt: String = "2026-05-20T10:00:00Z",
    ) = TranscriptEntity(
        id = id,
        workspaceId = "ws_1",
        audioPath = "/audio/$id.mp3",
        audioBasename = "$id.mp3",
        durationSeconds = 100.0,
        language = "en",
        createdAt = createdAt,
        jsonMtime = 1716200000.0,
        modelsAsr = null,
        modelsDiarizer = null,
        rawJson = "{}",
    )

    private fun seg(
        transcriptId: String,
        index: Int,
        start: Double,
        end: Double,
        speakerId: String?,
        text: String,
    ) = SegmentEntity(
        transcriptId = transcriptId,
        segmentIndex = index,
        startSec = start,
        endSec = end,
        text = text,
        speakerId = speakerId,
    )
}
