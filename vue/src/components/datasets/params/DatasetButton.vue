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
  name: "DatasetButton",
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
    createInterval() {
      this.interval = setTimeout(async () => {
        const data = await this.$store.dispatch("datasets/choiceProgress", {});
        const { finished, message, percent, data: dataset } = data;
        if (!data || finished) {
          this.$store.dispatch("messages/setProgressMessage", message);
          this.$store.dispatch("messages/setProgress", percent);
          if (data) {
            this.$store.dispatch('messages/setMessage', { message: `Датасет «${data.alias}» выбран`}, { root: true })
            this.$store.dispatch('projects/setProject', { dataset }, { root: true })
          }   
        } else {
          this.$store.dispatch("messages/setProgress", percent);
          this.$store.dispatch("messages/setProgressMessage", message);
          this.createInterval();
        }
        console.log(data);
      }, 1000);
    },
    async click(name) {
      if (name === "prepare") {
        const { alias, group, name } = this.selected;
        this.$store.dispatch("messages/setMessage", {
          message: `Выбран датасет «${name}»`,
        });
        const data = await this.$store.dispatch("datasets/choice", { alias, group });
        if (data) {
          this.createInterval();
        }
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