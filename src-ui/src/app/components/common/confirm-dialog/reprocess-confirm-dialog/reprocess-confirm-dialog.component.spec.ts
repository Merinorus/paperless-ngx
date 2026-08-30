import { provideHttpClient, withInterceptorsFromDi } from '@angular/common/http'
import { provideHttpClientTesting } from '@angular/common/http/testing'
import { ComponentFixture, TestBed } from '@angular/core/testing'
import { NgbActiveModal } from '@ng-bootstrap/ng-bootstrap'
import { RemoteOCRModeConfig } from 'src/app/data/paperless-config'
import { SETTINGS_KEYS } from 'src/app/data/ui-settings'
import { SettingsService } from 'src/app/services/settings.service'
import { ReprocessConfirmDialogComponent } from './reprocess-confirm-dialog.component'

describe('ReprocessConfirmDialogComponent', () => {
  let component: ReprocessConfirmDialogComponent
  let fixture: ComponentFixture<ReprocessConfirmDialogComponent>
  let settingsService: SettingsService

  const createComponent = (configured: boolean, mode: string) => {
    settingsService.set(SETTINGS_KEYS.REMOTE_OCR_CONFIGURED, configured)
    settingsService.set(SETTINGS_KEYS.REMOTE_OCR_MODE, mode)

    fixture = TestBed.createComponent(ReprocessConfirmDialogComponent)
    component = fixture.componentInstance
    fixture.detectChanges()
  }

  beforeEach(async () => {
    TestBed.configureTestingModule({
      providers: [
        NgbActiveModal,
        provideHttpClient(withInterceptorsFromDi()),
        provideHttpClientTesting(),
      ],
      imports: [ReprocessConfirmDialogComponent],
    }).compileComponents()

    settingsService = TestBed.inject(SettingsService)
  })

  it('should follow configured settings by default', () => {
    createComponent(true, RemoteOCRModeConfig.WORKFLOW_ONLY)

    expect(component.remoteOcrMode).toBe('configured')
  })

  it('should not offer remote OCR when no engine is configured', () => {
    createComponent(false, RemoteOCRModeConfig.WORKFLOW_ONLY)

    expect(component.showRemoteOcr).toBeFalsy()
    expect(
      fixture.nativeElement.querySelector('#reprocessRemoteOcrConfigured')
    ).toBeNull()
  })

  it('should offer overrides when remote OCR handles every document', () => {
    createComponent(true, RemoteOCRModeConfig.ALWAYS)

    expect(component.showRemoteOcr).toBeTruthy()
    expect(
      fixture.nativeElement.querySelector('#reprocessRemoteOcrConfigured')
    ).not.toBeNull()
  })

  it('should offer local, configured, and remote choices', () => {
    createComponent(true, RemoteOCRModeConfig.WORKFLOW_ONLY)

    expect(component.showRemoteOcr).toBeTruthy()
    expect(
      fixture.nativeElement.querySelector('#reprocessRemoteOcrLocal')
    ).not.toBeNull()
    expect(
      fixture.nativeElement.querySelector('#reprocessRemoteOcrConfigured')
    ).not.toBeNull()
    const remote = fixture.nativeElement.querySelector(
      '#reprocessRemoteOcrRemote'
    )
    expect(remote).not.toBeNull()

    remote.click()
    fixture.detectChanges()
    expect(component.remoteOcrMode).toBe('remote')
  })
})
