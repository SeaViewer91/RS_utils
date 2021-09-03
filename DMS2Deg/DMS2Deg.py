import PySimpleGUI as sg


def main():
    # Main Window의 Layout
    Layout = [
        [sg.Button('DMS to Deg', size=(20, 5), font=15), sg.Button('Deg to DMS', size=(20, 5), font=15)],
        [sg.Text('v1.0 by S.H.Bak')]
    ]

    # DMS to Deg 기능의 Layout
    LayoutDMS2Deg = [
        [sg.Text('Deg : '), sg.InputText(key='-getDeg-')],
        [sg.Text('Min : '), sg.InputText(key='-getMin-')],
        [sg.Text('Sec : '), sg.InputText(key='-getSec-')],
        [sg.Button('Transform'), sg.InputText(key='-DispDeg-')]
    ]

    # Deg to DMS 기능의 Layout
    LayoutDeg2DMS = [
        [sg.Text('Degree : '), sg.InputText(key='-getDeg-'), sg.Button('Transform')],
        [sg.Text('Deg : '), sg.InputText(key='-DispDeg-')],
        [sg.Text('Min : '), sg.InputText(key='-DispMin-')],
        [sg.Text('Sec : '), sg.InputText(key='-DispSec-')]
    ]

    MainWindow = sg.Window('DMS to Deg / Deg to DMS', Layout, size=(420, 140))
    DMS2DegWindow = sg.Window('DMS to Deg', LayoutDMS2Deg)
    Deg2DMSWindow = sg.Window('Deg to DMS', LayoutDeg2DMS)

    while True:
        MainEvent, MainValues = MainWindow.read()

        if MainEvent == sg.WIN_CLOSED:
            break

        if MainEvent == 'DMS to Deg':
            DMS2DegEvent, DMS2DegValues = DMS2DegWindow.read()

            if DMS2DegEvent == 'Transform':
                Deg = float(DMS2DegValues['-getDeg-'])
                Min = float(DMS2DegValues['-getMin-'])
                Sec = float(DMS2DegValues['-getSec-'])
                DMS2DegValues['-DispDeg-'] = Deg + Min / 60 + Sec / 3600

                DMS2DegWindow['-DispDeg-'].update(DMS2DegValues['-DispDeg-'])

        if MainEvent == 'Deg to DMS':
            Deg2DMSEvent, Deg2DMSValues = Deg2DMSWindow.read()

            if Deg2DMSEvent == 'Transform':
                Deg = float(Deg2DMSValues['-getDeg-'])
                DegIntPart = int(Deg)
                DegDecPart = Deg - DegIntPart

                Min = DegDecPart * 60
                MinIntPart = int(Min)
                MinDecPart = Min - MinIntPart
                print(MinDecPart)

                Sec = (Deg - DegIntPart - MinIntPart / 60) * 3600

                Deg2DMSValues['-DispDeg-'] = DegIntPart
                Deg2DMSValues['-DispMin-'] = MinIntPart
                Deg2DMSValues['-DispSec-'] = Sec

                Deg2DMSWindow['-DispDeg-'].update(Deg2DMSValues['-DispDeg-'])
                Deg2DMSWindow['-DispMin-'].update(Deg2DMSValues['-DispMin-'])
                Deg2DMSWindow['-DispSec-'].update(Deg2DMSValues['-DispSec-'])


if __name__ == '__main__':
    main()