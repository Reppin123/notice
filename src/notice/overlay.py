"""Transparent, click-through, always-on-top window over the main screen hosting overlay.html.

CSS pixels == screen points (top-left origin), so the page can draw at screen coordinates.
"""
from __future__ import annotations

from AppKit import (
    NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSColor,
    NSScreen, NSWindow, NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
)
from Foundation import NSURL, NSURLRequest
from WebKit import WKWebView, WKWebViewConfiguration

SCREEN_SAVER_LEVEL = 1000


def run_overlay(url: str):
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    frame = NSScreen.mainScreen().frame()
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.clearColor())
    win.setIgnoresMouseEvents_(True)
    win.setHasShadow_(False)
    win.setLevel_(SCREEN_SAVER_LEVEL)
    win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary)
    web = WKWebView.alloc().initWithFrame_configuration_(win.contentView().bounds(), WKWebViewConfiguration.alloc().init())
    web.setValue_forKey_(False, "drawsBackground")
    web.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(url)))
    win.setContentView_(web)
    win.orderFrontRegardless()
    app.run()
