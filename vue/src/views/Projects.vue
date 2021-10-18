<template>
  <main class="page-projects">
    <div class="wrapper">
      <h2>Мои проекты</h2>
      <NewModalCreateProject 
        :dialog="dialogCreate" 
        :loading="loading" 
        @create="createProject" 
        @close="dialogCreate = false" 
      />
      <NewModalDeleteProject 
        :dialog="dialogDelete" 
        :loading="loading" 
        @delete="deleteProject" 
        @close="dialogDelete = false" 
      />
      <div class="projects">
        <CardCreateProject @click.native="dialogCreate = true"/>
        <CardProject 
          v-bind="project" 
          v-for="(project, i) in projects" 
          :key="project.headline + i" 
          @deleteProject="dialogDelete = true" 
          @click.native="handleClick(project)" 
        />
      </div>
    </div>
  </main>
</template>

<script>
import CardProject from '@/components/projects/CardProject'
import NewModalCreateProject from '@/components/projects/modals/NewModalCreateProject'
import NewModalDeleteProject from '@/components/projects/modals/NewModalDeleteProject'
import CardCreateProject from '@/components/projects/CardCreateProject'
export default {
  name: 'Projects',
  components:{
    CardProject,
    CardCreateProject,
    NewModalCreateProject,
    NewModalDeleteProject
  },
  data: () => ({
    dialogCreate: false,
    dialogDelete:false,
    loading: false,
    selectProject: {},
    projects: [
      {
        id: 1,
        image: '', 
        active: false,
        created:'17 апреля 2021', 
        edited: '3 дня назад', 
        headline: 'Проект 1. Название максимум одна ст'
      },
      {
        id: 2,
        image: '', 
        active: false,
        created:'17 апреля 2021', 
        edited: '3 дня назад', 
        headline: 'Проект 1. Название максимум одна ст'
      },
      {
        id: 3,
        image: '', 
        active: false,
        created:'17 апреля 2021', 
        edited: '3 дня назад', 
        headline: 'Проект 1. Название максимум одна ст'
      }
    ]
  }),
  methods:{
    createProject(project){
      console.log('Create project', project)
    },
    deleteProject(){
      console.log('Delete project',this.selectProject)
    },
    handleClick(project){
      this.selectProject = project
      this.projects = this.projects.map(el => {
        return {
          ...el,
          active: el.id === project.id ? true : false
        }
      })
    }
  }
};
</script>

<style lang="scss" scoped>

.wrapper{
  background: #17212B;
  border-top: 0;
  height: 100%;
  padding: 20px;
  h2{
    margin-bottom: 20px;
    font-family: "Open Sans", sans-serif;
    font-size: 14px;
    font-weight: normal;
  }
}

.projects{
  display: grid;
  grid-template-columns:repeat(auto-fill, 300px);
  grid-gap: 30px;
}

</style>
