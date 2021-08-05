<template>
  <div>
    <button
      v-for="({ name, title }, i) of buttons"
      :key="i"
      :disabled="!selected"
      @click="click(name)"
    >
      {{ title }}
    </button>
  </div>
</template>

<script>
export default {
  name: 'DatasetButton',
  props: {
    buttons: {
      type: Array,
      default: () => [
        { name: "prepare", title: "Выбрать датасет", disabled: true },
        // { name: "delete", title: "Удалить", disabled: true },
        // { name: "change", title: "Редактировать", disabled: true },
      ],
    },
  },
  computed: {
    selected() {
      return this.$store.getters["datasets/getSelected"];
    },
  },
  methods: {
    async click(name) {
      if (name === "prepare") {
        const { alias, group, name } = this.selected;
        this.$store.dispatch("messages/setMessage", {
          message: `Выбран датасет «${name}»`,
        });
        await this.$store.dispatch("datasets/choice", { alias, group });
      }
    },
  },
};
</script>

<style lang="scss" scoped>
button {
  font-size: 0.875rem;
}
</style>