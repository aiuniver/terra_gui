<template>
  <div class="panel-settings">
    <div class="flex align-center">
      <t-field inline label="Ширина">
        <d-input-number v-model="width" />
      </t-field>
      <t-field inline label="Высота">
        <d-input-number v-model="height" />
      </t-field>
    </div>
    <div style="width: 100%">
      <t-field inline label="Сеть">
        <d-input-text v-model="network" />
      </t-field>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DatasetSettings',
  data: () => ({
    height: 128,
    width: 128,
    network: 'Сверточная',
  }),
};
</script>
