package app.locallexis.design

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Parchment = Color(0xFFF2EBDC)
private val ParchmentDim = Color(0xFFE4DCC7)
private val Ink = Color(0xFF1A1815)
private val InkSoft = Color(0xFF3D362C)
private val Accent = Color(0xFF7A5C2E)
private val AccentSoft = Color(0xFF9C7E4C)
private val Rule = Color(0xFF8A7B5C)

private val LightColors = lightColorScheme(
    primary = Accent,
    onPrimary = Parchment,
    background = Parchment,
    onBackground = Ink,
    surface = Parchment,
    onSurface = Ink,
    surfaceVariant = ParchmentDim,
    onSurfaceVariant = InkSoft,
    outline = Rule,
    outlineVariant = Rule,
)

private val DarkColors = darkColorScheme(
    primary = AccentSoft,
    onPrimary = Ink,
    background = Ink,
    onBackground = Parchment,
    surface = Ink,
    onSurface = Parchment,
    surfaceVariant = InkSoft,
    onSurfaceVariant = ParchmentDim,
    outline = ParchmentDim,
    outlineVariant = ParchmentDim,
)

@Composable
fun LocalLexisTheme(content: @Composable () -> Unit) {
    val colors = if (isSystemInDarkTheme()) DarkColors else LightColors
    MaterialTheme(colorScheme = colors, content = content)
}
