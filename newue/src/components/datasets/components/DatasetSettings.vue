<template>
  <div class="panel-settings">
    <div class="flex align-center">
      <TField inline label="Ширина">
        <DInputNumber v-model="width" />
      </TField>
      <TField inline label="Высота">
        <DInputNumber v-model="height" />
      </TField>
    </div>
    <div style="width: 100%">
      <TField inline label="Сеть">
        <DInputText v-model="network" />
      </TField>
    </div>
  </div>
</template>

<script>
import DInputNumber from '@/components/global/forms/components/DInputNumber';
import DInputText from '@/components/global/forms/components/DInputText';
import TField from '@/components/global/forms/components/TField';
export default {
  components: { TField, DInputNumber, DInputText },
  name: 'DatasetSettings',
  data: () => ({
    height: 128,
    width: 128,
    network: 'Сверточная',
  }),
};
</script>
