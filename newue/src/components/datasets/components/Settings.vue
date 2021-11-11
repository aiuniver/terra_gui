<template>
  <div class="panel-settings">
      <div class="flex align-center">
        <TField inline label="Ширина">
            <DInputNumber v-model="width" />
        </TField>
        <TField inline label="Высота">
            <DInputNumber v-model="height"/>
        </TField>
      </div>
      <div style="width:100%;">
          <TField inline label="Сеть">
              <DInputText v-model="network" />
          </TField>
      </div>
  </div>
</template>

<script>
import DInputNumber from '../../global/forms/components/DInputNumber.vue'
import DInputText from '../../global/forms/components/DInputText.vue'
import TField from '../../global/forms/components/TField.vue'
export default {
  components: { TField, DInputNumber, DInputText },
    name:"Settings",
    data: () => ({
        height: 128,
        width: 128,
        network: 'Сверточная'
    })
}
</script>
