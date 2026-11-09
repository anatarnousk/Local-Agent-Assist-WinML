using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;

namespace FabrikamAgentAssist;

public partial class App : Application
{
    private Window? _window;

    public App()
    {
        InitializeComponent();
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        DisablePowerThrottling();
        _window = new MainWindow();
        _window.Activate();
    }

    // Keep timings stable on stage: opt out of EcoQoS so Windows doesn't throttle the process when it isn't focused.
    private static void DisablePowerThrottling()
    {
        var state = new PowerThrottlingState { Version = 1, ControlMask = 0x1, StateMask = 0 };
        SetProcessInformation(GetCurrentProcess(), 4 /* ProcessPowerThrottling */, ref state, (uint)Marshal.SizeOf<PowerThrottlingState>());
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PowerThrottlingState { public uint Version, ControlMask, StateMask; }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentProcess();

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetProcessInformation(IntPtr hProcess, int infoClass, ref PowerThrottlingState info, uint size);
}
