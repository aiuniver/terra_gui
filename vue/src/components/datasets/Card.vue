<template>
  <div class="dataset-card-item">
    <div class="dataset-card">
      <div class="card-title">{{ name }}</div>
      <div class="card-body">
        <div v-for="(tag, key) of tags" :key="key" class="card-tag">
          {{ tag }}
        </div>
      </div>
      <div class="card-extra">
        <div class="wrapper">
          <span>{{ size ? size : 'Предустановленный' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    name: {
      type: String,
      default: "",
    },
    date: {
      type: String,
      default: "",
    },
    size: {
      type: String,
      default: null,
    },
    tags: {
      type: Object,
      default: () => {},
    },
  },
};
</script>
