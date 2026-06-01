package com.aifencingcoach.runtime

import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidManifestTest {
    @Test
    fun declaresInternetPermissionForGeminiApiCalls() {
        val manifest = File("src/main/AndroidManifest.xml")
        assertTrue("AndroidManifest.xml should exist for manifest permission checks.", manifest.exists())

        val document = DocumentBuilderFactory.newInstance()
            .apply { isNamespaceAware = true }
            .newDocumentBuilder()
            .parse(manifest)
        val permissions = document.getElementsByTagName("uses-permission")

        val hasInternetPermission = (0 until permissions.length).any { index ->
            val attributes = permissions.item(index).attributes
            attributes
                ?.getNamedItemNS("http://schemas.android.com/apk/res/android", "name")
                ?.nodeValue == "android.permission.INTERNET"
        }

        assertTrue(
            "Gemini summary requests need android.permission.INTERNET in AndroidManifest.xml.",
            hasInternetPermission
        )
    }
}
