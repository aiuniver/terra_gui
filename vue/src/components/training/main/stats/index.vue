<template>
  <div class="t-scatters">
    <div class="t-scatters__header">
      <div class="checks">
        <t-checkbox :inline="true" label="Выход “10”" @change="change" name="out10"/>
        <t-checkbox :inline="true" label="Выход “11”" @change="change" name="out11"/>
        <t-checkbox :inline="true" label="Автоотбновление"/>
      </div>
      <button @click="showContent = !showContent">Показать</button>
    </div>
    <div v-if="showContent" class="t-scatters__content">
      <Matrix v-if="out10" label="Выход “10”" />
      <Matrix v-if="out11" label="Выход “11”" />
    </div>
  </div>
</template>

<script>
import Matrix from './Matrix.vue'

export default {
  name: 't-scatters',
  components: {
      Matrix
  },
  data: () => ({
      out10: false,
      out11: false,
      showContent: false
  }),
  methods: {
      change(e) {
          this[e.name] = e.value
      }
  }
}
</script>

<style lang="scss" scoped>
.t-scatters {
  &__header {
    display: flex;
    .checks {
      display: flex;
      flex-wrap: wrap;
      flex-shrink: 0;
      max-width: 320px;
      * {
        flex: 0 0 150px;
      }
    }
    button {
      flex: 0 0 150px;
    }
  }
  &__content {
      display: flex;
      flex-wrap: wrap;
      gap: 50px;
  }
}
</style>