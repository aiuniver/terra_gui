<template>
  <div>
    <d-dropdown>
      <template #activator="{ on }">
        <d-input-text v-on="on" />
      </template>
      <ul>
        <li v-for="item of 10" :key="item">{{ 'dsdsdsd' + item }}</li>
      </ul>
    </d-dropdown>
  </div>
</template>

<script>
export default {
  name: 'v-select',
};
</script>