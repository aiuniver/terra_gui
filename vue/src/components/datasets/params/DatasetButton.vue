<template>
  <div class="params-item dataset-change pa-5">
    <div class="actions-form">
      <div v-for="({ name, title, disabled }, i) of buttons" :key="i">
        <button :disabled="disabled" @click.prevent="$emit('click', name)">
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
        { name: "prepare", title: "Подготовить", disabled: false },
        { name: "delete", title: "Удалить", disabled: true },
        { name: "change", title: "Редактировать", disabled: true },
      ],
    },
  },
};
</script>

<style lang="scss" scoped>
button{
  font-size: .875rem;
}
</style>