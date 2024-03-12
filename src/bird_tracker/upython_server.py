from machine import ADC

adc = ADC(0)

adc_buffer = [100]

def populate_buffer():
    for i, _ in enumerate(adc_buffer):
        adc_buffer[i] = adc.read()
        time.
