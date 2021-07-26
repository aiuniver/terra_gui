<template>
  <div class="params-item dataset-change pa-5">
    <div class="actions-form">
      <div v-for="({ name, title }, i) of buttons" :key="i">
        <button :disabled="!selected" @click="click(name)">
          {{ title }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    buttons: {
      type: Array,
      default: () => [
        { name: "prepare", title: "Подготовить", disabled: true },
        // { name: "delete", title: "Удалить", disabled: true },
        // { name: "change", title: "Редактировать", disabled: true },
      ],
    },
  },
  computed: {
    selected() {
      return this.$store.getters['datasets/getSelected']
    }
  },
  methods: {
    async click(name){
      if(name === 'prepare') {
        const { alias, group } = this.selected
        await this.$store.dispatch('datasets/choice', { alias, group })
      }
    }
  }
};
</script>

<style lang="scss" scoped>
button{
  font-size: .875rem;
}
</style>