<template>
  <nav class="menu">
    <ul class="flexbox-left-nowrap">
      <router-link to="/datasets" v-slot="{ isActive,  href, navigate }">
        <li :class="isActive ? 'active item' : 'item'">
          <a :href="href" @click="navigate">Данные</a>
        </li>
      </router-link>
      <router-link to="/modeling" v-slot="{ isActive,  href, navigate }">
        <li :class="isActive ? 'active item' : 'item'">
          <a :href="href" @click="navigate">Проектирование</a>
        </li>
      </router-link>
      <router-link to="/training" v-slot="{ isActive,  href, navigate }">
        <li :class="isActive ? 'active item' : 'item'">
          <a :href="href" @click="navigate">Обучение</a>
        </li>
      </router-link>
    </ul>
  </nav>
</template>

<script>
export default {
  created() {
    console.log(this.$router.history.current.fullPath === "/datasets");
  },
};
</script>